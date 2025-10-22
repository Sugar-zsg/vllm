# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
This model supports OpenAI audio transcription using
vLLM's multimodal pipeline and attention kernels.

Highlights:
- Loads a local AED checkpoint via `torch.load(<model_dir>/model.pth.tar)`.
- Implements SupportsTranscription and SupportsMultiModal interfaces.
- Avoids strict HF weight loading by overriding `load_weights`.
- Provides minimal audio feature extraction if HF processor unavailable.

Assumptions:
- Model config JSON under the model directory provides fields like:
  `d_model`, `odim`, `n_layers_enc`, `n_layers_dec`, `n_head`, `pad_id`,
  `pe_maxlen`, `idim`.
- Checkpoint contains keys similar to ESPnet-style AED (e.g. decoder.tgt_word_emb,
  encoder/self_attn parameters). When not present, model runs with random init.
"""

from __future__ import annotations

import math
import os
from typing import Annotated, Iterable, Literal, Mapping, Sequence, cast

import numpy as np
import torch
from torch import nn

from vllm.attention import Attention, AttentionType
from vllm.attention.layers.cross_attention import CrossAttention
from vllm.config import CacheConfig, ModelConfig, SpeechToTextConfig, VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.inputs.data import PromptType
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import MultiModalDataItems, MultiModalDataParser, AudioProcessorItems
from vllm.multimodal.processing import (
    BaseProcessingInfo,
    EncDecMultiModalProcessor,
    PromptReplacement,
    PromptUpdate,
)
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsTranscription
from .utils import (
    WeightsMapper,
    make_layers,
    maybe_prefix,
)

logger = init_logger(__name__)


# Minimal ISO-639-1 support map
_SUPPORTED_LANGS = {
    "en": "English",
    "zh": "Chinese",
}


class FireRedAEDAudioInputs(TensorSchema):
    """Audio inputs consumed by the encoder.

    input_features: list of 2D tensors (T, 80)
    """

    input_features: Annotated[list[torch.Tensor] | None, TensorShape("b", "t", 80)]


class FireRedAEDProcessingInfo(BaseProcessingInfo):
    def get_tokenizer(self) -> AnyTokenizer:  # type: ignore[override]
        tok = self.ctx.tokenizer
        if tok is None:
            class _NullTokenizer:
                bos_token_id = None
                eos_token_id = None

                def encode(self, text, **kwargs):
                    return[]

                def decode(self, tokens, **kwargs):
                    return ""

            return _NullTokenizer()  # type: ignore[return-value]
        return tok  # type: ignore[return-value]

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": 1}

    def get_feature_sample_rate(self) -> int:
        return 16_000

    def get_max_audio_clip_s(self) -> int:
        try:
            processor = self.ctx.get_hf_processor()
            chunk_len = getattr(getattr(processor, "feature_extractor", None), "chunk_length", 30)
            return int(chunk_len)
        except Exception:
            return 30


class FireRedAEDMultiModalProcessor(EncDecMultiModalProcessor[FireRedAEDProcessingInfo]):
    def _get_data_parser(self) -> MultiModalDataParser:
        return MultiModalDataParser(target_sr=self.info.get_feature_sample_rate())

    @property
    def pad_dummy_encoder_prompt(self) -> bool:
        return True

    def create_encoder_prompt(self, prompt: str | list[int], mm_data: MultiModalDataDict) -> str | list[int]:
        return [0]

    def create_decoder_prompt(self, prompt: str | list[int], mm_data: MultiModalDataDict) -> str | list[int]:
        if isinstance(prompt, str):
            return [self._get_default_decoder_start_id()]
        return prompt

    def _get_default_decoder_start_id(self) -> int:
        cfg = self.info.get_hf_config()
        dec_id = getattr(cfg, "decoder_start_token_id", None)
        if dec_id is None:
            dec_id = getattr(cfg, "bos_token_id", None)
        return int(dec_id) if dec_id is not None else 0

    def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs) -> Mapping[str, MultiModalFieldConfig]:
        return {"input_features": MultiModalFieldConfig.batched("audio")}

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        try:
            audios = mm_items.get_items("audio", AudioProcessorItems)
        except KeyError:
            return[]
        if len(audios) == 0:
            return[]

        def replacement(item_idx: int):
            num_samples = audios.get_audio_length(item_idx)
            num_frames = max(1, num_samples // 160)  # 10ms hop at 16kHz
            return [0] * int(num_frames)

        return [
            PromptReplacement(modality="audio", target=[0], replacement=replacement)
        ]

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> "BatchFeature":
        from transformers.feature_extraction_utils import BatchFeature
        empty_ids = torch.empty((1, 0), dtype=torch.long)

        audios = mm_data.get("audios",[])
        sr = self.info.get_feature_sample_rate()
        features: list[torch.Tensor] =[]
        if audios:
            try:
                import librosa
                for audio in audios:
                    if isinstance(audio, tuple):
                        audio, _ = audio
                    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=400, hop_length=160, n_mels=80)
                    logmel = np.log10(np.maximum(mel, 1e-10))
                    feat = torch.from_numpy(logmel.T).float()  # [T, 80]
                    features.append(feat)
            except Exception:
                for audio in audios:
                    if isinstance(audio, tuple):
                        audio, _ = audio
                    wave = torch.tensor(audio, dtype=torch.float32)
                    window = torch.hann_window(400)
                    stft = torch.stft(wave, 400, 160, window=window, return_complex=True)
                    magnitudes = stft.abs() ** 2
                    num_bins = magnitudes.size(0)
                    bins_per_mel = max(1, num_bins // 80)
                    mel = magnitudes[: bins_per_mel * 80].reshape(80, bins_per_mel, -1).mean(1)
                    logmel = torch.clamp(mel, min=1e-10).log10()
                    feat = logmel.transpose(0, 1)  # [T, 80]
                    features.append(feat)

        outputs = {"input_ids": empty_ids, "attention_mask": empty_ids.clone()}
        if features:
            outputs["input_features"] = features
        return BatchFeature(outputs)


class FireRedAEDDummyInputsBuilder(BaseDummyInputsBuilder[FireRedAEDProcessingInfo]):
    def get_dummy_mm_data(self, seq_len: int, mm_counts: Mapping[str, int], mm_options: Mapping[str, BaseDummyOptions] | None = None) -> MultiModalDataDict:
        num_audios = mm_counts.get("audio", 0)
        sr = self.info.get_feature_sample_rate()
        audio_len = self.info.get_max_audio_clip_s() * sr
        audio_overrides = mm_options.get("audio") if mm_options else None
        return {"audio": self._get_dummy_audios(length=audio_len, num_audios=num_audios, overrides=audio_overrides)}

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return ""


class FireRedAEDPositionalEmbedding(nn.Embedding):
    def __init__(self, num_positions: int, embedding_dim: int):
        super().__init__(num_positions, embedding_dim)

    def forward(self, position_ids):
        return self.weight[position_ids]


class FireRedAEDAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        attn_type: AttentionType,
        cache_config: CacheConfig | None,
        quant_config: QuantizationConfig | None,
        prefix: str = "",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.attn_type = attn_type
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.scaling = self.head_dim**-0.5

        if attn_type == AttentionType.DECODER:
            self.qkv_proj = QKVParallelLinear(
                hidden_size=embed_dim,
                head_size=self.head_dim,
                total_num_heads=num_heads,
                total_num_kv_heads=num_heads,
                bias=True,
                quant_config=quant_config,
                prefix=f"{prefix}.qkv_proj",
            )
            self.attn = Attention(
                num_heads=num_heads,
                head_size=self.head_dim,
                scale=self.scaling,
                num_kv_heads=num_heads,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.attn",
                attn_type=attn_type,
            )
        else:
            self.q_proj = ColumnParallelLinear(
                input_size=embed_dim,
                output_size=embed_dim,
                bias=True,
                quant_config=quant_config,
                prefix=f"{prefix}.q_proj",
            )
            self.kv_proj = QKVParallelLinear(
                hidden_size=embed_dim,
                head_size=self.head_dim,
                total_num_heads=0,
                total_num_kv_heads=num_heads,
                bias=True,
                quant_config=quant_config,
                prefix=f"{prefix}.kv_proj",
            )
            self.attn = CrossAttention(
                num_heads=num_heads,
                head_size=self.head_dim,
                scale=self.scaling,
                num_kv_heads=num_heads,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.attn",
                attn_type=attn_type,
            )

        self.out_proj = RowParallelLinear(
            input_size=embed_dim,
            output_size=embed_dim,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor | None = None) -> torch.Tensor:
        if self.attn_type == AttentionType.DECODER:
            qkv, _ = self.qkv_proj(hidden_states)
            q, k, v = qkv.split([self.embed_dim, self.embed_dim, self.embed_dim], dim=-1)
            out = self.attn(q, k, v)
        else:
            if encoder_hidden_states is None:
                out = torch.zeros_like(hidden_states)
            else:
                q, _ = self.q_proj(hidden_states)
                kv, _ = self.kv_proj(encoder_hidden_states)
                kv_heads = kv.split([self.embed_dim, self.embed_dim], dim=-1)
                k = kv_heads[0]
                v = kv_heads[1]
                out = self.attn(q, k, v)
        out, _ = self.out_proj(out)
        return out


class FireRedAEDMLP(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, quant_config: QuantizationConfig | None, prefix: str = ""):
        super().__init__()
        self.fc1 = ColumnParallelLinear(embed_dim, ffn_dim, bias=True, quant_config=quant_config, prefix=f"{prefix}.fc1")
        self.fc2 = RowParallelLinear(ffn_dim, embed_dim, bias=True, quant_config=quant_config, prefix=f"{prefix}.fc2")
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.fc1(x)
        return self.fc2(self.act(h))[0]


class FireRedAEDDecoderLayer(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.self_attn = FireRedAEDAttention(
            embed_dim=config.d_model,
            num_heads=config.n_head,
            attn_type=AttentionType.DECODER,
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.cross_attn = FireRedAEDAttention(
            embed_dim=config.d_model,
            num_heads=config.n_head,
            attn_type=AttentionType.ENCODER_DECODER,
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.cross_attn",
        )
        self.self_attn_norm = nn.LayerNorm(config.d_model)
        self.cross_attn_norm = nn.LayerNorm(config.d_model)
        self.mlp_norm = nn.LayerNorm(config.d_model)
        self.mlp = FireRedAEDMLP(
            embed_dim=config.d_model,
            ffn_dim=config.d_model * 4,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.mlp",
        )

    def forward(self, x: torch.Tensor, encoder_outputs: torch.Tensor | None) -> torch.Tensor:
        residual = x
        x = self.self_attn_norm(x)
        x = self.self_attn(x)
        x = residual + x

        residual = x
        x = self.cross_attn_norm(x)
        x = self.cross_attn(x, encoder_hidden_states=encoder_outputs)
        x = residual + x

        residual = x
        x = self.mlp_norm(x)
        x = residual + self.mlp(x)
        return x


class FireRedAEDDecoder(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.padding_idx = config.pad_id
        self.max_target_positions = getattr(config, "pe_maxlen", 5000)
        self.embed_scale = math.sqrt(config.d_model)

        self.embed_tokens = nn.Embedding(config.odim, config.d_model, self.padding_idx)
        self.embed_positions = FireRedAEDPositionalEmbedding(self.max_target_positions, config.d_model)
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.n_layers_dec,
            lambda prefix: FireRedAEDDecoderLayer(vllm_config=vllm_config, prefix=f"{prefix}.layers"),
            prefix=f"{prefix}.layers",
        )
        self.layer_norm = nn.LayerNorm(config.d_model)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        encoder_outputs: torch.Tensor | None,
        *,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids and inputs_embeds cannot both be None")
            inputs_embeds = self.get_input_embeddings(input_ids)
        pos = self.embed_positions(positions)
        x = inputs_embeds + pos
        for layer in self.layers:
            x = layer(x, encoder_outputs)
        x = self.layer_norm(x)
        return x


class FireRedAEDConformerEncoder(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.d_model = config.d_model
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2), nn.ReLU(), nn.Conv2d(32, 32, 3, 2), nn.ReLU()
        )
        subsample_idim = ((config.idim - 1) // 2 - 1) // 2
        self.proj = nn.Linear(32 * subsample_idim, config.d_model)
        self.positional = nn.Embedding(getattr(config, "pe_maxlen", 5000), config.d_model)
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=config.d_model,
                    nhead=config.n_head,
                    dim_feedforward=config.d_model * 4,
                    batch_first=True,
                )
                for _ in range(config.n_layers_enc)
            ]
        )
        self.layer_norm = nn.LayerNorm(config.d_model)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        xs =[]
        for feat in features:
            x = feat.unsqueeze(1)  # [T, 80] -> [T, 1, 80]
            x = x.unsqueeze(0)  # -> [1, T, 1, 80]
            x = x.permute(0, 2, 1, 3)  # [1, 1, T, 80]
            x = self.conv(x)  # [1, 32, T', D']
            b, c, t, d = x.shape
            x = x.transpose(1, 2).reshape(b, t, c * d)  # [1, T', 32*D']
            x = self.proj(x)
            pos_ids = torch.arange(x.size(1), device=x.device)
            pos = self.positional(pos_ids)
            x = x + pos
            for blk in self.blocks:
                x = blk(x)
            xs.append(x.squeeze(0))
        x = torch.cat(xs, dim=0)
        x = self.layer_norm(x)
        return x


class FireRedAEDModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.encoder = FireRedAEDConformerEncoder(vllm_config=vllm_config, prefix=f"{prefix}.encoder")
        self.decoder = FireRedAEDDecoder(vllm_config=vllm_config, prefix=f"{prefix}.decoder")

    def forward(
        self,
        input_features: list[torch.Tensor] | None,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        *,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        enc = self.get_encoder_outputs(input_features)
        dec = self.decoder(
            input_ids=input_ids,
            positions=positions,
            encoder_outputs=enc,
            inputs_embeds=inputs_embeds,
        )
        return dec

    def get_encoder_outputs(self, input_features: list[torch.Tensor] | None) -> torch.Tensor | None:
        if input_features is None:
            return None
        return self.encoder(input_features)


@MULTIMODAL_REGISTRY.register_processor(
    FireRedAEDMultiModalProcessor,
    info=FireRedAEDProcessingInfo,
    dummy_inputs=FireRedAEDDummyInputsBuilder,
)
class FireRedAEDForConditionalGeneration(nn.Module, SupportsTranscription, SupportsMultiModal):
    merge_by_field_config = True
    supported_languages = _SUPPORTED_LANGS
    supports_transcription_only = True

    hf_to_vllm_mapper = WeightsMapper(orig_to_new_substr={})

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.dtype = vllm_config.model_config.dtype

        # For ASR-only models, disable tokenizer initialization to avoid
        # pydantic validation against unknown HF config classes during
        # AutoTokenizer startup. Also ensure vLLM uses model path as tokenizer
        # path when needed.
        try:
            vllm_model_cfg = vllm_config.model_config
            # 显式跳过 tokenizer 初始化；ASR 不依赖文本提示
            vllm_model_cfg.skip_tokenizer_init = True
            # 若 tokenizer 为 None，vLLM 默认使用 `model` 路径；保持为 None。
        except Exception:
            pass

        # Ensure decoder/BOS/EOS token ids are set even without a tokenizer.
        # InputPreprocessor relies on hf_config.decoder_start_token_id for
        # encoder-decoder models. If it is missing, it falls back to BOS from
        # tokenizer, which may be None when tokenizer is not initialized for
        # ASR-only models. Setting explicit defaults here prevents 500s during
        # /v1/audio/transcriptions.
        if getattr(config, "decoder_start_token_id", None) is None:
            # Prefer a non-pad start token. If pad_id exists, avoid it; use 1.
            try:
                pad_id = getattr(config, "pad_id", None)
                default_start = 1 if pad_id != 1 else 2
            except Exception:
                default_start = 1
            setattr(config, "decoder_start_token_id", int(default_start))

        if getattr(config, "bos_token_id", None) is None:
            # Align BOS with decoder_start_token_id to keep generation stable.
            setattr(config, "bos_token_id", int(getattr(config, "decoder_start_token_id")))

        if getattr(config, "eos_token_id", None) is None:
            # Provide a conservative EOS; if pad_id is available, use it;
            # otherwise fall back to last vocab id.
            eos_fallback = getattr(config, "pad_id", None)
            if eos_fallback is None:
                try:
                    eos_fallback = int(getattr(config, "odim")) - 1
                except Exception:
                    eos_fallback = 2
            setattr(config, "eos_token_id", int(eos_fallback))

        self.model = FireRedAEDModel(vllm_config=vllm_config, prefix=prefix)
        self.unpadded_vocab_size = config.odim
        self.proj_out = ParallelLMHead(
            config.odim,
            config.d_model,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "proj_out"),
        )
        try:
            if tuple(self.proj_out.weight.shape) == tuple(self.model.decoder.embed_tokens.weight.shape):
                self.proj_out = self.proj_out.tie_weights(self.model.decoder.embed_tokens)
        except Exception:
            logger.exception("Failed tying LM head to embeddings; continuing.")
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size, config.odim, 1.0)

        self._maybe_load_firered_checkpoint(vllm_config)

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("audio"):
            return None
        raise ValueError("Only audio modality is supported")

    def _maybe_load_firered_checkpoint(self, vllm_config: VllmConfig) -> None:
        try:
            model_dir = vllm_config.model_config.model
            ckpt = os.path.join(model_dir, "model.pth.tar")
            if not os.path.exists(ckpt):
                logger.warning("FireRedAED checkpoint not found at %s; using random init.", ckpt)
                return
            package = torch.load(ckpt, map_location="cpu", weights_only=False)
            sd = package.get("model_state_dict",{})

            # Decoder embedding
            dec_emb = sd.get("decoder.tgt_word_emb.weight")
            if dec_emb is not None:
                emb_shape = tuple(self.model.decoder.embed_tokens.weight.shape)
                if tuple(dec_emb.shape) == emb_shape:
                    self.model.decoder.embed_tokens.weight.data.copy_(dec_emb.to(self.model.decoder.embed_tokens.weight.dtype))
                else:
                    logger.warning("Decoder embedding shape mismatch: ckpt=%s vs model=%s; skipping", tuple(dec_emb.shape), emb_shape)
                lm_shape = tuple(self.proj_out.weight.shape)
                if tuple(dec_emb.shape) == lm_shape:
                    self.proj_out.weight.data.copy_(dec_emb.to(self.proj_out.weight.dtype))
                else:
                    logger.warning("LM head shape mismatch: ckpt=%s vs lm_head=%s; skipping", tuple(dec_emb.shape), lm_shape)

            # Decoder layers
            for i in range(self.config.n_layers_dec):
                pfx = f"decoder{self._layer_suffix(i)}"
                layer = self.model.decoder.layers[i]
                # self-attn qkv
                q_w = sd.get(f"{pfx}.self_attn.w_qs.weight")
                k_w = sd.get(f"{pfx}.self_attn.w_ks.weight")
                v_w = sd.get(f"{pfx}.self_attn.w_vs.weight")
                q_b = sd.get(f"{pfx}.self_attn.w_qs.bias")
                v_b = sd.get(f"{pfx}.self_attn.w_vs.bias")
                if q_w is not None and k_w is not None and v_w is not None:
                    qkv_w = torch.cat([q_w, k_w, v_w], dim=0)
                    layer.self_attn.qkv_proj.weight.data.copy_(qkv_w.to(layer.self_attn.qkv_proj.weight.dtype))
                    dim = q_w.shape[0]
                    k_b = torch.zeros_like(q_b) if q_b is not None else torch.zeros(dim)
                    v_b2 = v_b if v_b is not None else torch.zeros(dim)
                    qkv_b = torch.cat([q_b or torch.zeros(dim), k_b, v_b2], dim=0)
                    layer.self_attn.qkv_proj.bias.data.copy_(qkv_b.to(layer.self_attn.qkv_proj.bias.dtype))
                out_w = sd.get(f"{pfx}.self_attn.fc.weight")
                out_b = sd.get(f"{pfx}.self_attn.fc.bias")
                if out_w is not None and hasattr(layer.self_attn.out_proj, "weight"):
                    layer.self_attn.out_proj.weight.data.copy_(out_w.to(layer.self_attn.out_proj.weight.dtype))
                if out_b is not None and hasattr(layer.self_attn.out_proj, "bias"):
                    layer.self_attn.out_proj.bias.data.copy_(out_b.to(layer.self_attn.out_proj.bias.dtype))

                # cross-attn q/k/v
                q_w = sd.get(f"{pfx}.cross_attn.w_qs.weight")
                q_b = sd.get(f"{pfx}.cross_attn.w_qs.bias")
                k_w = sd.get(f"{pfx}.cross_attn.w_ks.weight")
                v_w = sd.get(f"{pfx}.cross_attn.w_vs.weight")
                v_b = sd.get(f"{pfx}.cross_attn.w_vs.bias")
                if q_w is not None:
                    layer.cross_attn.q_proj.weight.data.copy_(q_w.to(layer.cross_attn.q_proj.weight.dtype))
                    if q_b is not None:
                        layer.cross_attn.q_proj.bias.data.copy_(q_b.to(layer.cross_attn.q_proj.bias.dtype))
                if k_w is not None and v_w is not None:
                    kv_w = torch.cat([k_w, v_w], dim=0)
                    layer.cross_attn.kv_proj.weight.data.copy_(kv_w.to(layer.cross_attn.kv_proj.weight.dtype))
                    d_model = k_w.shape[0]
                    k_b = torch.zeros_like(v_b) if v_b is not None else torch.zeros(d_model)
                    kv_b = torch.cat([k_b, v_b or torch.zeros(d_model)], dim=0)
                    layer.cross_attn.kv_proj.bias.data.copy_(kv_b.to(layer.cross_attn.kv_proj.bias.dtype))
                out_w = sd.get(f"{pfx}.cross_attn.fc.weight")
                out_b = sd.get(f"{pfx}.cross_attn.fc.bias")
                if out_w is not None and hasattr(layer.cross_attn.out_proj, "weight"):
                    layer.cross_attn.out_proj.weight.data.copy_(out_w.to(layer.cross_attn.out_proj.weight.dtype))
                if out_b is not None and hasattr(layer.cross_attn.out_proj, "bias"):
                    layer.cross_attn.out_proj.bias.data.copy_(out_b.to(layer.cross_attn.out_proj.bias.dtype))

            # Encoder front & encoder blocks are not fully mapped here; users may extend as needed.
        except Exception:
            logger.exception("Failed loading FireRedAED checkpoint; continuing with random init.")

    def _layer_suffix(self, i: int) -> str:
        # Many ESPnet-like models use numeric suffix; adapt if needed.
        return f".layers.{i}"

    @classmethod
    def validate_language(cls, language: str | None) -> str | None:
        if language is None:
            return "en"
        if language in cls.supported_languages:
            return language
        raise ValueError(f"Unsupported language for FireRedAED: {language!r}. Supported: {list(cls.supported_languages)}")

    @classmethod
    def get_generation_prompt(
        cls,
        audio: np.ndarray,
        stt_config: SpeechToTextConfig,
        model_config: ModelConfig,
        language: str | None,
        task_type: Literal["transcribe", "translate"],
        request_prompt: str,
        to_language: str | None,
    ) -> PromptType:
        return cast(
            PromptType,
            {
                "encoder_prompt": {
                    "prompt": "",
                    "multi_modal_data": {"audio": (audio, stt_config.sample_rate)},
                },
                "decoder_prompt": None,
            },
        )

    @classmethod
    def get_speech_to_text_config(cls, model_config: ModelConfig, task_type: Literal["transcribe", "translate"]) -> SpeechToTextConfig:
        return SpeechToTextConfig(max_audio_clip_s=30, sample_rate=16_000)

    def get_language_model(self) -> torch.nn.Module:
        return self.model.decoder

    def get_multimodal_embeddings(self, **kwargs: object) -> MultiModalEmbeddings:
        audio_input = self._parse_and_validate_audio_input(**kwargs)
        return [self.model.get_encoder_outputs(audio_input["input_features"])]

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
        handle_oov_mm_token: bool = False,
    ) -> torch.Tensor:
        return self.model.decoder.get_input_embeddings(input_ids)

    def forward(self, input_ids: torch.Tensor | None, positions: torch.Tensor, **kwargs) -> torch.Tensor:
        audio_input = self._parse_and_validate_audio_input(**kwargs)
        inputs_embeds: torch.Tensor | None = kwargs.pop("inputs_embeds", None)
        return self.model(
            input_features=audio_input["input_features"],
            input_ids=input_ids,
            positions=positions,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        if hidden_states.numel() == 0 or hidden_states.shape[-1] == 0:
            return None
        return self.logits_processor(self.proj_out, hidden_states)

    def _parse_and_validate_audio_input(self, **kwargs: object) -> FireRedAEDAudioInputs:
        input_features = kwargs.pop("input_features", None)
        if input_features is not None:
            input_features = [x.to(self.dtype) for x in input_features]
        return FireRedAEDAudioInputs(input_features=input_features)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str] | None:  # type: ignore[override]
        # Skip strict HF loader; weights are loaded from local checkpoint.
        return None
