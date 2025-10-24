# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Inference-only FireRedASR LLM adapter for vLLM.

This adapter follows the qwen2_audio/whisper patterns:
- Accepts raw audio via the multimodal interface
- Replaces a single "<speech>" token in the prompt with an equal-length
  sequence of feature placeholders
- Computes speech embeddings via an external encoder+adapter, then merges
  them into input embeddings at the placeholder positions

Notes
- This file provides a minimal scaffolding so we can iterate locally.
- Actual model weights loading is deferred to user-provided assets.
- For now we keep dependencies optional and raise clear errors if missing.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    AudioItem,
    ModalityData,
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    AudioProcessorItems,
    AudioEmbeddingItems,
    DictEmbeddingItems,
    ModalityDataItems,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from transformers import BatchFeature
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
from .utils import AutoWeightsLoader, init_vllm_registered_model, maybe_prefix


# Token used by the FireRedASR chat template to indicate audio placement
_DEFAULT_SPEECH_TOKEN_TEXT = "<speech>"


class FireRedASRAudioEmbeddingInputs(TensorSchema):
    """
    Audio embeddings input type.

    Dimensions:
        - bn: Batch size
        - naf: Number of audio features (time frames after downsample)
        - hs: Hidden size (must match the hidden size of language model)
    """

    type: Literal["audio_embeds"] = "audio_embeds"

    audio_embeds: Annotated[
        list[torch.Tensor],
        TensorShape("bn", "naf", "hs"),
    ]


FireRedASRInputs = FireRedASRAudioEmbeddingInputs


class FireRedASRProcessingInfo(BaseProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        # One audio per prompt is typical; lift if multi-audio is supported.
        return {"audio": 1}

    # Provide a minimal dummy HF processor to satisfy engine expectations.
    def get_hf_processor(self, **kwargs: object) -> "ProcessorMixin":  # type: ignore[name-defined]
        from transformers.processing_utils import ProcessorMixin
        class _DummyProcessor(ProcessorMixin):  # type: ignore[misc]
            def __init__(self, tokenizer):
                self._tokenizer = tokenizer
            def __call__(self, *, text: str = "", return_tensors: str = "pt", **_: object):
                from transformers import BatchFeature
                prompt_ids = self._tokenizer.encode(text, add_special_tokens=False)
                return BatchFeature(dict(input_ids=[prompt_ids]), tensor_type=return_tensors)

        return _DummyProcessor(self.get_tokenizer())


class FireRedASRDummyInputsBuilder(BaseDummyInputsBuilder[FireRedASRProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)
        # Insert one <speech> for each audio
        return _DEFAULT_SPEECH_TOKEN_TEXT * num_audios

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, Any] | None = None,
    ) -> MultiModalDataDict:
        num_audios = mm_counts.get("audio", 0)
        # Produce dummy audio embeddings directly (no raw audio).
        # Choose small lengths to keep startup fast.
        naf = 12  # number of frames
        hs = 1024  # hidden size (LLM embed dim; arbitrary for dummy)
        embeds = [torch.zeros(naf, hs) for _ in range(num_audios)]
        return {"audio": {"audio_embeds": embeds}}


def _field_config_for_embeddings(hf_inputs: Mapping[str, torch.Tensor]):
    # Accept only audio_embeds path (no HF processor fields)
    return dict(audio_embeds=MultiModalFieldConfig.batched("audio"))


class FireRedASRDataParser(MultiModalDataParser):
    def _parse_audio_data(
        self,
        data: dict[str, torch.Tensor] | ModalityData[AudioItem],
    ) -> ModalityDataItems[Any, Any] | None:
        # Gracefully handle text-only or startup/dummy paths.
        if data is None:
            return AudioProcessorItems(None)
        if self._is_empty(data):
            return None

        # If user provides precomputed embeddings
        if isinstance(data, dict):
            return DictEmbeddingItems(
                data,
                modality="audio",
                required_fields={"audio_embeds"},
                fields_factory=_field_config_for_embeddings,
            )
        # If embedding tensors passed directly
        if self._is_embeddings(data):
            return AudioEmbeddingItems(data)  # type: ignore[name-defined]

        # For any other input type (raw audio), treat as absent to avoid startup failures.
        return AudioProcessorItems(None)


class FireRedASRMultiModalProcessor(BaseMultiModalProcessor[FireRedASRProcessingInfo]):
    def _get_data_parser(self) -> MultiModalDataParser:
        # No resampling enforced here; upstream can pass (audio, sr) tuples.
        return FireRedASRDataParser()

    def _get_mm_fields_config(
        self,
        hf_inputs: Mapping[str, Any],
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        # If user passed audio_embeds through kwargs → route directly
        return _field_config_for_embeddings(hf_inputs)

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        """Bypass HF Processor and tokenize prompt directly.

        FireRedASR adapter does not rely on a HuggingFace Processor; we only
        need the tokenized prompt IDs. Audio embeddings are passed through
        separately and do not require HF processing here.
        """
        tokenizer = self.info.get_tokenizer()
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        return BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        # Try to derive the number of embeddings per audio from precomputed
        # `audio_embeds` (if provided).
        out_mm_data = out_mm_kwargs.get_data()
        audio_embeds = out_mm_data.get("audio_embeds")

        tokenizer = self.info.get_tokenizer()
        pad_id = getattr(tokenizer, "pad_token_id", None)
        if pad_id is None:
            # Fallback to vocab if pad token id missing
            pad_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")

        def replacement_for_item(item_idx: int):
            if audio_embeds is None:
                # Unknown length: at minimum replace `<speech>` with one pad,
                # the merge step will fail if counts mismatch, which is better
                # than silently producing wrong outputs.
                return PromptUpdateDetails.select_token_id([pad_id], embed_token_id=pad_id)

            num_features = audio_embeds[item_idx].shape[0]
            tokens = [pad_id] * num_features
            return PromptUpdateDetails.select_token_id(tokens, embed_token_id=pad_id)

        # Match the literal text `<speech>` in the prompt and replace it with
        # a sequence of pad tokens of equal length to the audio embeddings.
        return [
            PromptReplacement(
                modality="audio",
                target=_DEFAULT_SPEECH_TOKEN_TEXT,
                replacement=replacement_for_item,
            )
        ]


@MULTIMODAL_REGISTRY.register_processor(
    FireRedASRMultiModalProcessor,
    info=FireRedASRProcessingInfo,
    dummy_inputs=FireRedASRDummyInputsBuilder,
)
class FireRedASRLLMForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsPP):
    """
    FireRedASR-LLM adapter module.

    This class owns only the ASR encoder/projector and the LLM wrapper from
    vLLM; the language model is constructed via `init_vllm_registered_model`.
    """

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("audio"):
            # Chat-template uses a single <speech> token
            return _DEFAULT_SPEECH_TOKEN_TEXT
        raise ValueError("Only audio modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config

        # Initialize language model using text sub-config if provided; otherwise
        # defer to config of the target LLM in HF.
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config={
                "_name_or_path": "/workspace/bella-infra/user/zhangshuge002/FireRedASR/FireRedASR/pretrained_models/Qwen2-7B-Instruct",
                "architectures": ["Qwen2ForCausalLM"],
                "attention_dropout": 0,
                "bos_token_id": 151643,
                "eos_token_id": 151645,
                "hidden_act": "silu",
                "hidden_size": 3584,
                "initializer_range": 0.02,
                "intermediate_size": 18944,
                "max_position_embeddings": 32768,
                "max_window_layers": 28,
                "model_type": "qwen2",
                "num_attention_heads": 28,
                "num_hidden_layers": 28,
                "num_key_value_heads": 4,
                "rms_norm_eps": 0.000001,
                "rope_theta": 1000000,
                 "sliding_window": 131072,
                 "tie_word_embeddings": False,
                 "torch_dtype": "bfloat16",
                 "transformers_version": "4.41.2",
                 "use_cache": True,
                 "use_sliding_window": False,
                 "vocab_size": 152064
            },
            prefix=maybe_prefix(prefix, "language_model"),
        )

        # FireRedASR-specific: try to import encoder+adapter from project code.
        # We keep this optional to unblock iterative development.
        self.encoder = None
        self.adapter = None
        self.hidden_size = getattr(getattr(self.config, "text_config", self.config), "hidden_size", None)

        try:
            # Prefer local development repo if present
            import importlib
            import sys
            from pathlib import Path

            dev_repo = Path("/workspace/bella-infra/user/zhangshuge002/FireRedASR/FireRedASR")
            packaged_repo = Path(self.vllm_config.model_config.model)  # e.g., .../fireredasr-llm-hf
            if dev_repo.exists() and str(dev_repo) not in sys.path:
                sys.path.insert(0, str(dev_repo))
            if packaged_repo.exists() and str(packaged_repo) not in sys.path:
                sys.path.insert(0, str(packaged_repo))

            # Lazy imports from FireRedASR
            firered_models = importlib.import_module("models.fireredasr_llm")
            adapter_mod = importlib.import_module("models.module.adapter")
            asr_aed_mod = importlib.import_module("models.fireredasr_aed")

            # Load encoder from packaged checkpoint if provided
            # Attempt: <model_dir>/asr_encoder.pth.tar; if missing, fall back to
            # loading the full package and taking its encoder attribute.
            import os
            model_dir = str(self.vllm_config.model_config.model)
            asr_encoder_path = os.path.join(model_dir, "asr_encoder.pth.tar")

            if hasattr(firered_models, "FireRedAsrLlm"):
                FireRedAsrLlm = getattr(firered_models, "FireRedAsrLlm")
                Adapter = getattr(adapter_mod, "Adapter")
                if os.path.exists(asr_encoder_path):
                    # Load only the encoder from AED checkpoint
                    def _load_encoder(path: str):
                        pkg = torch.load(path, map_location="cpu")
                        FireRedAsrAed = getattr(asr_aed_mod, "FireRedAsrAed")
                        model = FireRedAsrAed.from_args(pkg["args"])  # type: ignore[index]
                        if "model_state_dict" in pkg:
                            model.load_state_dict(pkg["model_state_dict"], strict=False)
                        return model.encoder, model.encoder.odim

                    encoder, encoder_dim = _load_encoder(asr_encoder_path)
                else:
                    # Fallback: try to reconstruct FireRedAsrLlm fully and reuse its encoder
                    full_ckpt = os.path.join(model_dir, "model.pth.tar")
                    if not os.path.exists(full_ckpt):
                        raise FileNotFoundError(
                            f"Missing FireRedASR checkpoints under: {model_dir}"
                        )
                    pkg = torch.load(full_ckpt, map_location="cpu")
                    args = pkg["args"]
                    # Provide llm_dir if present in config
                    if getattr(self.config, "llm_dir", None):
                        args.llm_dir = self.config.llm_dir
                    model = FireRedAsrLlm.from_args(args)
                    model.load_state_dict(pkg["model_state_dict"], strict=False)
                    encoder = model.encoder
                    encoder_dim = encoder.odim

                # Build adapter to LLM hidden size
                if self.hidden_size is None:
                    raise ValueError("Cannot infer LLM hidden_size from config")
                downsample = int(getattr(self.config, "encoder_downsample_rate", 2))
                self.adapter = Adapter(encoder_dim, self.hidden_size, downsample)
                self.encoder = encoder
        except Exception:
            # Leave encoder/adapter as None; we can still accept precomputed embeds
            pass

        # Expose methods for pipeline parallel/vLLM interface
        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors

    # --------------- vLLM multimodal hooks ---------------
    def _run_encoder(self, audio: torch.Tensor, lengths: torch.Tensor) -> list[torch.Tensor]:
        if self.encoder is None or self.adapter is None:
            raise RuntimeError(
                "FireRedASR encoder/adapter not initialized. "
                "Pass precomputed `audio_embeds`, or ensure FireRedASR code and checkpoints are available."
            )

        self.encoder.eval()
        with torch.no_grad():
            enc_outs, enc_lengths, _ = self.encoder(audio, lengths)
            # Adapter returns (features, new_lengths)
            speech_features, new_lens = self.adapter(enc_outs, enc_lengths)

        # Split batch into a list of [Ti, hidden]
        outputs: list[torch.Tensor] =[]
        for i, t in enumerate(new_lens.tolist()):
            outputs.append(speech_features[i, :t])
        return outputs

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def get_multimodal_embeddings(self, **kwargs: object) -> MultiModalEmbeddings:
        # Accept either precomputed audio_embeds or raw audio features + lengths
        audio_embeds: torch.Tensor | list[torch.Tensor] | None = kwargs.pop("audio_embeds", None)
        if audio_embeds is not None:
            if isinstance(audio_embeds, torch.Tensor) and audio_embeds.ndim == 3:
                # [B, T, H] -> split into list
                return [t for t in audio_embeds]
            if isinstance(audio_embeds, list):
                return audio_embeds  # already list of [T, H]
            raise ValueError("Invalid `audio_embeds` format; expect [B,T,H] or list of [T,H]")

        # Raw features path: expect `(feats, lengths)` already prepared.
        # Upstream parser produces only NumPy/tensors; to keep the adapter simple,
        # we require users to prepare framed features and lengths externally or via
        # their FireRedASR utilities when running locally.
        input_features = kwargs.get("input_features")
        feature_lengths = kwargs.get("feature_lengths") or kwargs.get("feature_attention_mask")

        if input_features is None or feature_lengths is None:
            # Nothing to embed
            return[]

        if not isinstance(input_features, torch.Tensor):
            input_features = torch.as_tensor(input_features)
        if not isinstance(feature_lengths, torch.Tensor):
            feature_lengths = torch.as_tensor(feature_lengths)
        if feature_lengths.ndim > 1:
            feature_lengths = feature_lengths.sum(-1)

        return self._run_encoder(input_features, feature_lengths)

    # --------------- vLLM forward/logits ---------------
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None
        return self.language_model.model(
            input_ids, positions, intermediate_tensors, inputs_embeds=inputs_embeds
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    # --------------- Weights loading ---------------
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
