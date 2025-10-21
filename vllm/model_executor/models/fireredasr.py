# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Native vLLM wrapper for FireRedASR (AED), enabling OpenAI audio transcription
via vLLM’s scheduling and attention kernels.

This integrates an encoder that consumes audio features and a vLLM-style
decoder layer with flash/page attention. It implements SupportsTranscription
and SupportsMultiModal to work with vLLM's OpenAI audio routes.

Notes:
- We optionally map weights from a packaged checkpoint at
  `<model_path>/model.pth.tar` where possible.
- Audio features are derived from raw audio via the multimodal pipeline.
  We estimate frame count for prompt replacement using a 10ms hop.
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
from vllm.model_executor.layers.activation import get_act_fn
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
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsTranscription
from .utils import (
    WeightsMapper,
    make_layers,
    maybe_prefix,
)

logger = init_logger(__name__)


# Minimal ISO-639-1 support map (extend as needed)
_SUPPORTED_LANGS = {
    "en": "English",
    "zh": "Chinese",
}


class FireRedAudioInputs(TensorSchema):
    """Audio inputs consumed by the encoder.

    input_features: list of 2D tensors (T, 80)
    """

    input_features: Annotated[list[torch.Tensor] | None, TensorShape("b", "t", 80)]


class FireRedASRProcessingInfo(BaseProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": 1}

    def get_feature_sample_rate(self) -> int:
        # FireRedASR typically uses 16kHz
        return 16_000

    def get_max_audio_clip_s(self) -> int:
        # Fallback to 30s if HF processor is not available
        try:
            processor = self.ctx.get_hf_processor()
            chunk_len = getattr(getattr(processor, "feature_extractor", None), "chunk_length", 30)
            return int(chunk_len)
        except Exception:
            return 30


class FireRedASRMultiModalProcessor(EncDecMultiModalProcessor[FireRedASRProcessingInfo]):
    def _get_data_parser(self) -> MultiModalDataParser:
        return MultiModalDataParser(target_sr=self.info.get_feature_sample_rate())

    @property
    def pad_dummy_encoder_prompt(self) -> bool:
        return True

    def create_encoder_prompt(self, prompt: str | list[int], mm_data: MultiModalDataDict) -> str | list[int]:
        # Create a dummy encoder prompt (padded to the number of audio tokens) for profiling
        return [0]

    def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs) -> Mapping[str, MultiModalFieldConfig]:
        # We only feed audio features to encoder
        return dict(input_features=MultiModalFieldConfig.batched("audio"))

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        # Replace dummy token in encoder prompt with a sequence sized to audio frames
        audios = mm_items.get_items("audio", AudioProcessorItems)
        if len(audios) == 0:
            return[]

        # Estimate projector features per audio using 10ms hop at 16kHz: T ≈ samples/160
        def replacement(item_idx: int):
            # Estimate frames using 10ms hop at 16kHz
            num_samples = audios.get_audio_length(item_idx)
            num_frames = max(1, num_samples // 160)
            return [0] * int(num_frames)

        return [
            PromptReplacement(
                modality="audio",
                target=[0],
                replacement=replacement,
            )
        ]

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> "BatchFeature":
        # Minimal processor: tokenize decoder prompt and convert raw audio to mel features.
        from transformers.feature_extraction_utils import BatchFeature
        tokenizer = self.info.get_tokenizer()

        # Tokenize decoder prompt (text)
        input_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        attn_mask = [1] * len(input_ids)

        # Convert audio list -> log-mel features [T, 80]
        audios = mm_data.get("audios", [])
        sr = self.info.get_feature_sample_rate()
        features: list[torch.Tensor] =[]
        if audios:
            try:
                import librosa
                for (audio, orig_sr) in audios:  # vLLM parser returns (np.ndarray, sr)
                    if orig_sr != sr:
                        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)
                    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=400, hop_length=160, n_mels=80)
                    logmel = np.log10(np.maximum(mel, 1e-10))
                    # [80, T] -> [T, 80]
                    feat = torch.from_numpy(logmel.T).float()
                    features.append(feat)
            except Exception:
                # Fallback to torch STFT + mel filter
                for (audio, orig_sr) in audios:
                    wave = torch.tensor(audio, dtype=torch.float32)
                    if orig_sr != sr:
                        # naive resample using numpy.interp to avoid torch.interp
                        ratio = sr / float(orig_sr)
                        target_len = int(round(len(wave) * ratio))
                        t = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
                        src_t = np.linspace(0.0, 1.0, len(wave), dtype=np.float32)
                        wave = torch.from_numpy(np.interp(t, src_t, wave.cpu().numpy())).to(wave.dtype).to(wave.device)
                    window = torch.hann_window(400)
                    stft = torch.stft(wave, 400, 160, window=window, return_complex=True)
                    magnitudes = stft.abs() ** 2
                    # Simple mel filter approximation: average bins into 80 groups
                    num_bins = magnitudes.size(0)
                    bins_per_mel = max(1, num_bins // 80)
                    mel = magnitudes[: bins_per_mel * 80].reshape(80, bins_per_mel, -1).mean(1)
                    logmel = torch.clamp(mel, min=1e-10).log10()
                    feat = logmel.transpose(0, 1)  # [80, T] -> [T, 80]
                    features.append(feat)

        outputs = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn_mask, dtype=torch.long),
        }
        if features:
            outputs["input_features"] = features
        return BatchFeature(outputs)


class FireRedASRDummyInputsBuilder(BaseDummyInputsBuilder[FireRedASRProcessingInfo]):
    def get_dummy_mm_data(self, seq_len: int, mm_counts: Mapping[str, int], mm_options: Mapping[str, BaseDummyOptions] | None = None) -> MultiModalDataDict:
        num_audios = mm_counts.get("audio", 0)
        sr = self.info.get_feature_sample_rate()
        audio_len = self.info.get_max_audio_clip_s() * sr
        audio_overrides = mm_options.get("audio") if mm_options else None
        return {"audio": self._get_dummy_audios(length=audio_len, num_audios=num_audios, overrides=audio_overrides)}

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        # No special decoder tokens; start from empty prompt
        return ""


class FireRedPositionalEmbedding(nn.Embedding):
    def __init__(self, num_positions: int, embedding_dim: int):
        super().__init__(num_positions, embedding_dim)

    def forward(self, position_ids):
        return self.weight[position_ids]


class FireRedAttention(nn.Module):
    """Multi-headed attention layer leveraging vLLM kernels.

    Configured for either self-attention (decoder) or cross-attention.
    """

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

        # QKV projections: self-attn stacks qkv; cross-attn uses q_proj and kv_proj
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.attn_type == AttentionType.DECODER:
            qkv, _ = self.qkv_proj(hidden_states)
            q, k, v = qkv.split(
                [self.num_heads * self.head_dim, self.num_heads * self.head_dim, self.num_heads * self.head_dim],
                dim=-1,
            )
            attn_out = self.attn(q, k, v)
        else:
            q, _ = self.q_proj(hidden_states)
            kv, _ = self.kv_proj(encoder_hidden_states)
            k, v = kv.split(
                [self.num_heads * self.head_dim, self.num_heads * self.head_dim],
                dim=-1,
            )
            attn_out = self.attn(q, k, v)

        out, _ = self.out_proj(attn_out)
        return out


class FireRedMLP(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, act_fn: str, *, quant_config: QuantizationConfig | None, prefix: str):
        super().__init__()
        self.activation_fn = get_act_fn(act_fn)
        self.fc1 = ColumnParallelLinear(input_size=embed_dim, output_size=ffn_dim, quant_config=quant_config, prefix=f"{prefix}.fc1")
        self.fc2 = RowParallelLinear(input_size=ffn_dim, output_size=embed_dim, quant_config=quant_config, prefix=f"{prefix}.fc2")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.fc1(x)
        x = self.activation_fn(x)
        x, _ = self.fc2(x)
        return x


class FireRedDecoderLayer(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.self_attn = FireRedAttention(
            embed_dim=config.d_model,
            num_heads=config.n_head,
            attn_type=AttentionType.DECODER,
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.cross_attn = FireRedAttention(
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
        self.mlp = FireRedMLP(
            embed_dim=config.d_model,
            ffn_dim=config.d_model * 4,
            act_fn="gelu",
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


class FireRedDecoder(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.padding_idx = config.pad_id
        self.max_target_positions = getattr(config, "pe_maxlen", 5000)
        self.embed_scale = math.sqrt(config.d_model)

        self.embed_tokens = nn.Embedding(config.odim, config.d_model, self.padding_idx)
        self.embed_positions = FireRedPositionalEmbedding(self.max_target_positions, config.d_model)
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.n_layers_dec,
            lambda prefix: FireRedDecoderLayer(vllm_config=vllm_config, prefix=f"{prefix}.layers"),
            prefix=f"{prefix}.layers",
        )
        self.layer_norm = nn.LayerNorm(config.d_model)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, encoder_outputs: torch.Tensor | None) -> torch.Tensor:
        inputs_embeds = self.get_input_embeddings(input_ids)
        pos = self.embed_positions(positions)
        x = inputs_embeds + pos
        for layer in self.layers:
            x = layer(x, encoder_outputs)
        x = self.layer_norm(x)
        return x


class FireRedConformerEncoder(nn.Module):
    """Minimal integration of FireRedASR encoder frontend + blocks.

    For compatibility, we implement a lightweight conv subsampling front-end
    and a stack of transformer encoder layers to produce encoder outputs for
    the decoder cross-attention.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.d_model = config.d_model
        # Conv subsampling frontend
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


class FireRedASRModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.encoder = FireRedConformerEncoder(vllm_config=vllm_config, prefix=f"{prefix}.encoder")
        self.decoder = FireRedDecoder(vllm_config=vllm_config, prefix=f"{prefix}.decoder")

    def forward(self, input_features: list[torch.Tensor] | None, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        enc = self.get_encoder_outputs(input_features)
        dec = self.decoder(input_ids=input_ids, positions=positions, encoder_outputs=enc)
        return dec

    def get_encoder_outputs(self, input_features: list[torch.Tensor] | None) -> torch.Tensor | None:
        if input_features is None:
            return None
        return self.encoder(input_features)


@MULTIMODAL_REGISTRY.register_processor(
    FireRedASRMultiModalProcessor,
    info=FireRedASRProcessingInfo,
    dummy_inputs=FireRedASRDummyInputsBuilder,
)
class FireRedASRForConditionalGeneration(nn.Module, SupportsTranscription, SupportsMultiModal):
    """vLLM-native FireRedASR AED transcription model."""

    merge_by_field_config = True
    supported_languages = _SUPPORTED_LANGS
    supports_transcription_only = True

    # Map stacked QKV projections names to individual ones for weight remapping
    hf_to_vllm_mapper = WeightsMapper(orig_to_new_substr={})

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.dtype = vllm_config.model_config.dtype

        self.model = FireRedASRModel(vllm_config=vllm_config, prefix=prefix)
        self.unpadded_vocab_size = config.odim
        self.proj_out = ParallelLMHead(
            config.odim,
            config.d_model,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "proj_out"),
        )
        # Tie output proj to embedding
        self.proj_out = self.proj_out.tie_weights(self.model.decoder.embed_tokens)
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size, config.odim, 1.0)

        # Attempt to load FireRedASR checkpoint weights directly
        self._maybe_load_firered_checkpoint(vllm_config)

    def _maybe_load_firered_checkpoint(self, vllm_config: VllmConfig) -> None:
        try:
            model_dir = vllm_config.model_config.model
            ckpt = os.path.join(model_dir, "model.pth.tar")
            if not os.path.exists(ckpt):
                logger.warning("FireRedASR checkpoint not found at %s; using random init.", ckpt)
                return
            package = torch.load(ckpt, map_location="cpu", weights_only=False)
            sd = package.get("model_state_dict",{})
            # Load embeddings (decoder) and projection
            dec_emb = sd.get("decoder.tgt_word_emb.weight")
            if dec_emb is not None:
                self.model.decoder.embed_tokens.weight.data.copy_(dec_emb)
                self.proj_out.weight.data.copy_(dec_emb)
            # Load decoder norms
            for i in range(self.config.n_layers_dec):
                pfx = f"decoder.layer_stack.{i}"
                self._copy_if_exists(sd, f"{pfx}.self_attn_norm.weight", self.model.decoder.layers[i].self_attn_norm.weight)
                self._copy_if_exists(sd, f"{pfx}.self_attn_norm.bias", self.model.decoder.layers[i].self_attn_norm.bias)
                self._copy_if_exists(sd, f"{pfx}.cross_attn_norm.weight", self.model.decoder.layers[i].cross_attn_norm.weight)
                self._copy_if_exists(sd, f"{pfx}.cross_attn_norm.bias", self.model.decoder.layers[i].cross_attn_norm.bias)
                self._copy_if_exists(sd, f"{pfx}.mlp_norm.weight", self.model.decoder.layers[i].mlp_norm.weight)
                self._copy_if_exists(sd, f"{pfx}.mlp_norm.bias", self.model.decoder.layers[i].mlp_norm.bias)
                # MLP weights
                self._copy_if_exists(sd, f"{pfx}.mlp.w_1.weight", self._get_param(self.model.decoder.layers[i].mlp.fc1, "weight"))
                self._copy_if_exists(sd, f"{pfx}.mlp.w_1.bias", self._get_param(self.model.decoder.layers[i].mlp.fc1, "bias"))
                self._copy_if_exists(sd, f"{pfx}.mlp.w_2.weight", self._get_param(self.model.decoder.layers[i].mlp.fc2, "weight"))
                self._copy_if_exists(sd, f"{pfx}.mlp.w_2.bias", self._get_param(self.model.decoder.layers[i].mlp.fc2, "bias"))
                # Attention Q/K/V and out proj: best-effort mapping into QKV stacks
                self._remap_decoder_attn(sd, pfx, self.model.decoder.layers[i])
            # Encoder: front-end convs
            self._copy_if_exists(sd, "encoder.input_preprocessor.conv.0.weight", self._get_param(self.model.encoder.conv[0], "weight"))
            self._copy_if_exists(sd, "encoder.input_preprocessor.conv.0.bias", self._get_param(self.model.encoder.conv[0], "bias"))
            self._copy_if_exists(sd, "encoder.input_preprocessor.conv.2.weight", self._get_param(self.model.encoder.conv[2], "weight"))
            self._copy_if_exists(sd, "encoder.input_preprocessor.conv.2.bias", self._get_param(self.model.encoder.conv[2], "bias"))
        except Exception:
            logger.exception("Failed loading FireRedASR checkpoint; continuing with current parameters.")

    @staticmethod
    def _get_param(module: nn.Module, name: str) -> torch.Tensor | None:
        try:
            return getattr(module, name)
        except Exception:
            return None

    @staticmethod
    def _copy_if_exists(sd: dict[str, torch.Tensor], key: str, param: torch.Tensor | None) -> None:
        if param is None:
            return
        w = sd.get(key)
        if w is None:
            return
        if tuple(param.shape) == tuple(w.shape):
            param.data.copy_(w)

    def _remap_decoder_attn(self, sd: dict[str, torch.Tensor], pfx: str, layer: FireRedDecoderLayer) -> None:
        # Map FireRed separate q/k/v weights to stacked QKVParallelLinear
        q_w = sd.get(f"{pfx}.self_attn.w_qs.weight")
        k_w = sd.get(f"{pfx}.self_attn.w_ks.weight")
        v_w = sd.get(f"{pfx}.self_attn.w_vs.weight")
        q_b = sd.get(f"{pfx}.self_attn.w_qs.bias")
        v_b = sd.get(f"{pfx}.self_attn.w_vs.bias")
        # Create stacked qkv if available
        if q_w is not None and k_w is not None and v_w is not None:
            qkv_w = torch.cat([q_w, k_w, v_w], dim=0)
            layer.self_attn.qkv_proj.weight.data.copy_(qkv_w)
            # Bias: k has no bias in FireRed; initialize zeros
            d_model = q_w.shape[0]
            k_b = torch.zeros_like(q_b) if q_b is not None else torch.zeros(d_model)
            qkv_b = torch.cat([q_b or torch.zeros(d_model), k_b, v_b or torch.zeros(d_model)], dim=0)
            layer.self_attn.qkv_proj.bias.data.copy_(qkv_b)
        # out proj
        out_w = sd.get(f"{pfx}.self_attn.fc.weight")
        out_b = sd.get(f"{pfx}.self_attn.fc.bias")
        if out_w is not None and hasattr(layer.self_attn.out_proj, "weight"):
            layer.self_attn.out_proj.weight.data.copy_(out_w)
        if out_b is not None and hasattr(layer.self_attn.out_proj, "bias"):
            layer.self_attn.out_proj.bias.data.copy_(out_b)

        # Cross-attn: q/k/v
        q_w = sd.get(f"{pfx}.cross_attn.w_qs.weight")
        q_b = sd.get(f"{pfx}.cross_attn.w_qs.bias")
        k_w = sd.get(f"{pfx}.cross_attn.w_ks.weight")
        v_w = sd.get(f"{pfx}.cross_attn.w_vs.weight")
        v_b = sd.get(f"{pfx}.cross_attn.w_vs.bias")
        if q_w is not None:
            layer.cross_attn.q_proj.weight.data.copy_(q_w)
            if q_b is not None:
                layer.cross_attn.q_proj.bias.data.copy_(q_b)
        if k_w is not None and v_w is not None:
            kv_w = torch.cat([k_w, v_w], dim=0)
            layer.cross_attn.kv_proj.weight.data.copy_(kv_w)
            d_model = k_w.shape[0]
            k_b = torch.zeros_like(v_b) if v_b is not None else torch.zeros(d_model)
            kv_b = torch.cat([k_b, v_b or torch.zeros(d_model)], dim=0)
            layer.cross_attn.kv_proj.bias.data.copy_(kv_b)
        out_w = sd.get(f"{pfx}.cross_attn.fc.weight")
        out_b = sd.get(f"{pfx}.cross_attn.fc.bias")
        if out_w is not None and hasattr(layer.cross_attn.out_proj, "weight"):
            layer.cross_attn.out_proj.weight.data.copy_(out_w)
        if out_b is not None and hasattr(layer.cross_attn.out_proj, "bias"):
            layer.cross_attn.out_proj.bias.data.copy_(out_b)

    @classmethod
    def validate_language(cls, language: str | None) -> str | None:
        # Default to English if not provided
        if language is None:
            return "en"
        if language in cls.supported_languages:
            return language
        raise ValueError(
            f"Unsupported language for FireRedASR: {language!r}. Supported: {list(cls.supported_languages)}"
        )

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
        # Decoder starts from request prompt; encoder consumes audio features.
        return cast(
            PromptType,
            {
                "encoder_prompt": {
                    "prompt": "",
                    "multi_modal_data": {"audio": (audio, stt_config.sample_rate)},
                },
                "decoder_prompt": request_prompt or "",
            },
        )

    @classmethod
    def get_speech_to_text_config(
        cls, model_config: ModelConfig, task_type: Literal["transcribe", "translate"]
    ) -> SpeechToTextConfig:
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
        # Text-only decoder embeddings
        return self.model.decoder.get_input_embeddings(input_ids)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, **kwargs) -> torch.Tensor:
        audio_input = self._parse_and_validate_audio_input(**kwargs)
        return self.model(
            input_features=audio_input["input_features"],
            input_ids=input_ids,
            positions=positions,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.proj_out, hidden_states)

    def _parse_and_validate_audio_input(self, **kwargs: object) -> FireRedAudioInputs:
        input_features = kwargs.pop("input_features", None)
        if input_features is not None:
            input_features = [x.to(self.dtype) for x in input_features]
        return FireRedAudioInputs(input_features=input_features)