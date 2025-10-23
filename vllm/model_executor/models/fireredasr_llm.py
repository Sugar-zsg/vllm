# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""FireRedASR-LLM integration for vLLM.

This model adapts the FireRedASR encoder + adapter pipeline to vLLM's
multimodal interface and merges speech features into an underlying LLM.

Usage notes:
- Prompts should include a speech placeholder token. By default, we use
  "<speech>". The tokenizer must contain this token; if missing, we attempt
  to add it at runtime.
- Audio inputs are passed via the multimodal input API (e.g., audio arrays);
  this processor computes log-Mel features and expands the placeholder into
  the correct number of tokens.
- The underlying language model (e.g., Qwen2) is resolved from hf_config.text_config.
- Optionally, encoder weights can be loaded from a local PyTorch checkpoint
  (e.g., asr_encoder.pth.tar) via hf_config.asr_encoder_path.
"""

from __future__ import annotations

from typing import Annotated, Mapping

import math
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BatchFeature, PretrainedConfig

from vllm.config import CacheConfig, VllmConfig
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.interfaces import SupportsMultiModal, SupportsPP
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    init_vllm_registered_model,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig
from vllm.multimodal.parse import AudioProcessorItems, MultiModalDataParser
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
)
from vllm.sequence import IntermediateTensors
from vllm.utils.tensor_schema import TensorSchema, TensorShape


# Default speech placeholder token for prompts
DEFAULT_SPEECH_TOKEN = "<speech>"


# -------------------------------
# Minimal feature extractor
# -------------------------------

class _SimpleFbank:
    """Lightweight log-Mel filterbank extractor.

    Tries to use `torchaudio` when available; otherwise, falls back to a
    simple NumPy implementation. This is not intended to be a bit-for-bit
    match to the original Kaldi implementation, but suffices for inference
    integration.
    """

    def __init__(self, sample_rate: int = 16000, n_mels: int = 80):
        self.sample_rate = int(sample_rate)
        self.n_mels = int(n_mels)
        try:
            import torchaudio
            self._ta = torchaudio
            self._mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate, n_mels=self.n_mels
            )
        except Exception:
            self._ta = None
            self._mel = None

    def __call__(self, wav: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(wav, np.ndarray):
            wav_t = torch.from_numpy(wav).float()
        else:
            wav_t = wav.float()
        wav_t = wav_t.view(-1)
        if self._mel is not None:
            mel = self._mel(wav_t)
            mel = torch.log(mel + 1e-6)
            return mel.transpose(0, 1)  # (T, n_mels)
        # Fallback: naive framing + DFT-based mel approximation
        frame_len = int(self.sample_rate * 0.025)
        frame_shift = int(self.sample_rate * 0.010)
        n_fft = 1 << (frame_len - 1).bit_length()
        # Pad end
        num_frames = 1 + max(0, (len(wav_t) - frame_len) // frame_shift)
        frames =[]
        for i in range(num_frames):
            start = i * frame_shift
            end = start + frame_len
            if end > len(wav_t):
                pad = torch.zeros(end - len(wav_t))
                frame = torch.cat([wav_t[start:], pad])
            else:
                frame = wav_t[start:end]
            window = torch.hann_window(frame_len)
            frame = frame * window
            spec = torch.fft.rfft(frame, n_fft)
            power = (spec.real**2 + spec.imag**2)
            frames.append(power)
        spec = torch.stack(frames)  # (T, n_fft/2+1)
        # Simple triangular mel filterbank
        def _mel(f_hz: float) -> float:
            return 2595.0 * math.log10(1.0 + f_hz / 700.0)
        f_min, f_max = 0.0, self.sample_rate / 2
        m_min, m_max = _mel(f_min), _mel(f_max)
        m_points = torch.linspace(m_min, m_max, self.n_mels + 2)
        f_points = 700.0 * (10.0 ** (m_points / 2595.0) - 1.0)
        bins = torch.floor((n_fft + 1) * f_points / self.sample_rate).long()
        fb = torch.zeros(self.n_mels, n_fft // 2 + 1)
        for m in range(1, self.n_mels + 1):
            f_left, f_center, f_right = bins[m - 1], bins[m], bins[m + 1]
            if f_center == f_left:
                f_center += 1
            if f_right == f_center:
                f_right += 1
            fb[m - 1, f_left:f_center] = torch.linspace(0, 1, f_center - f_left)
            fb[m - 1, f_center:f_right] = torch.linspace(1, 0, f_right - f_center)
        mel = spec @ fb.T
        mel = torch.log(mel + 1e-6)
        return mel  # (T, n_mels)


class _ASRFeatExtractor:
    def __init__(self, sample_rate: int = 16000, n_mels: int = 80):
        self._fbank = _SimpleFbank(sample_rate=sample_rate, n_mels=n_mels)

    def __call__(self, audios: list[np.ndarray | torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        feats = [self._fbank(a) for a in audios]
        lengths = torch.tensor([f.shape[0] for f in feats], dtype=torch.long)
        max_len = int(lengths.max().item()) if len(lengths) > 0 else 0
        if max_len == 0:
            return torch.zeros((len(audios), 0, self._fbank.n_mels)), lengths
        padded = torch.zeros((len(audios), max_len, self._fbank.n_mels), dtype=feats[0].dtype)
        for i, f in enumerate(feats):
            padded[i, : f.shape[0]] = f
        return padded, lengths


# -------------------------------
# FireRedASR encoder + adapter
# -------------------------------

class _Adapter(nn.Module):
    def __init__(self, encoder_dim: int, llm_dim: int, downsample_rate: int = 2):
        super().__init__()
        self.ds = int(downsample_rate)
        self.linear1 = nn.Linear(encoder_dim * self.ds, llm_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(llm_dim, llm_dim)

    def forward(self, x: torch.Tensor, x_lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, feat_dim = x.size()
        num_frames_to_discard = seq_len % self.ds
        if num_frames_to_discard > 0:
            x = x[:, :-num_frames_to_discard, :]
            seq_len = x.size(1)
        x = x.contiguous().view(batch_size, seq_len // self.ds, feat_dim * self.ds)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        new_x_lens = torch.clamp(x_lens, max=seq_len) // self.ds
        return x, new_x_lens


# Conformer components (minimal subset used by FireRedASR encoder)

class _Conv2dSubsampling(nn.Module):
    def __init__(self, idim: int, d_model: int, out_channels: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, out_channels, 3, 2),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 2),
            nn.ReLU(),
        )
        subsample_idim = ((idim - 1) // 2 - 1) // 2
        self.out = nn.Linear(out_channels * subsample_idim, d_model)
        self.subsampling = 4
        self.context = 7  # left+1+right

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x.unsqueeze(1)
        x = self.conv(x)
        N, C, T, D = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(N, T, C * D))
        mask = x_mask[:, :, :-2:2][:, :, :-2:2]
        input_lengths = mask[:, -1, :].sum(dim=-1)
        return x, input_lengths, mask


class _RelPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe_positive = torch.zeros(max_len, d_model, requires_grad=False)
        pe_negative = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)).item() / d_model))
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)
        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Tmax, T = self.pe.size(1), x.size(1)
        pos_emb = self.pe[:, Tmax // 2 - T + 1 : Tmax // 2 + T].clone().detach()
        return pos_emb


class _ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature: float):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(0.0)
        self.INF = float("inf")

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        attn = torch.matmul(q, k.transpose(2, 3)) / self.temperature
        if mask is not None:
            m = mask.unsqueeze(1).eq(0)
            attn = attn.masked_fill(m, -self.INF)
            attn = torch.softmax(attn, dim=-1).masked_fill(m, 0.0)
        else:
            attn = torch.softmax(attn, dim=-1)
        d_attn = self.dropout(attn)
        output = torch.matmul(d_attn, v)
        return output, attn


class _EncoderMultiHeadAttention(nn.Module):
    def __init__(self, n_head: int, d_model: int, residual_dropout: float = 0.1):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.d_v = self.d_k
        self.w_qs = nn.Linear(d_model, n_head * self.d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * self.d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * self.d_v, bias=False)
        self.layer_norm_q = nn.LayerNorm(d_model)
        self.layer_norm_k = nn.LayerNorm(d_model)
        self.layer_norm_v = nn.LayerNorm(d_model)
        self.attention = _ScaledDotProductAttention(temperature=self.d_k**0.5)
        self.fc = nn.Linear(n_head * self.d_v, d_model, bias=False)
        self.dropout = nn.Dropout(residual_dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        sz_b, len_q = q.size(0), q.size(1)
        residual = q
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        # LN + projections
        q = self.layer_norm_q(q)
        k = self.layer_norm_k(k)
        v = self.layer_norm_v(v)
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k).transpose(1, 2)
        k = self.w_ks(k).view(sz_b, k.size(1), n_head, d_k).transpose(1, 2)
        v = self.w_vs(v).view(sz_b, v.size(1), n_head, d_v).transpose(1, 2)
        output, attn = self.attention(q, k, v, mask=mask)
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))
        output = output + residual
        return output, attn


class _ConformerConvModule(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 33, dropout_rate: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.depthwise_conv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, groups=d_model)
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        residual = x
        out = self.layer_norm(x)
        out = out.transpose(1, 2)
        out = self.pointwise_conv1(out)
        out = F.glu(out, dim=1)
        out = self.depthwise_conv(out)
        out = out.transpose(1, 2)
        out = F.silu(self.batch_norm(out.transpose(1, 2))).transpose(1, 2)
        out = self.dropout(self.pointwise_conv2(out.transpose(1, 2)))
        if mask is not None:
            out = out.transpose(1, 2)
            out.masked_fill_(mask.ne(1), 0.0)
            out = out.transpose(1, 2)
        return out + residual


class _ConformerFeedForward(nn.Module):
    def __init__(self, d_model: int, dropout_rate: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, 4 * d_model)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(4 * d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class _RelPosMultiHeadAttention(_EncoderMultiHeadAttention):
    def __init__(self, n_head: int, d_model: int, residual_dropout: float = 0.1):
        super().__init__(n_head, d_model, residual_dropout)
        d_k = d_model // n_head
        self.scale = 1.0 / (d_k**0.5)
        self.linear_pos = nn.Linear(d_model, n_head * d_k, bias=False)
        self.pos_bias_u = nn.Parameter(torch.FloatTensor(n_head, d_k))
        self.pos_bias_v = nn.Parameter(torch.FloatTensor(n_head, d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def _rel_shift(self, x: torch.Tensor) -> torch.Tensor:
        N, H, T1, T2 = x.size()
        zero_pad = torch.zeros((N, H, T1, 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1).view(N, H, T2 + 1, T1)
        x = x_padded[:, :, 1:].view_as(x)
        return x[:, :, :, : x.size(-1) // 2 + 1]

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, pos_emb: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        sz_b, len_q = q.size(0), q.size(1)
        residual = q
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        q = self.layer_norm_q(q)
        k = self.layer_norm_k(k)
        v = self.layer_norm_v(v)
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k).transpose(1, 2)
        k = self.w_ks(k).view(sz_b, k.size(1), n_head, d_k).transpose(1, 2)
        v = self.w_vs(v).view(sz_b, v.size(1), n_head, d_v).transpose(1, 2)
        # Relative position
        pos = self.linear_pos(pos_emb).view(sz_b, pos_emb.size(1), n_head, d_k).transpose(1, 2)
        q_with_bias_u = q + self.pos_bias_u.unsqueeze(0).unsqueeze(2)
        q_with_bias_v = q + self.pos_bias_v.unsqueeze(0).unsqueeze(2)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(2, 3))
        matrix_bd = torch.matmul(q_with_bias_v, pos.transpose(2, 3))
        matrix_bd = self._rel_shift(matrix_bd)
        attn = (matrix_ac + matrix_bd) * self.scale
        if mask is not None:
            m = mask.unsqueeze(1).eq(0)
            attn = attn.masked_fill(m, float("-inf"))
            attn = torch.softmax(attn, dim=-1).masked_fill(m, 0.0)
        else:
            attn = torch.softmax(attn, dim=-1)
        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))
        output = output + residual
        return output, attn


class _RelPosEmbConformerBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, residual_dropout: float = 0.1, dropout_rate: float = 0.1, kernel_size: int = 33):
        super().__init__()
        self.ffn1 = _ConformerFeedForward(d_model, dropout_rate)
        self.mhsa = _RelPosMultiHeadAttention(n_head, d_model, residual_dropout)
        self.conv = _ConformerConvModule(d_model, kernel_size, dropout_rate)
        self.ffn2 = _ConformerFeedForward(d_model, dropout_rate)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, pos_emb: torch.Tensor, slf_attn_mask: torch.Tensor | None = None, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        out = 0.5 * x + 0.5 * self.ffn1(x)
        out = self.mhsa(out, out, out, pos_emb, mask=slf_attn_mask)[0]
        out = self.conv(out, pad_mask)
        out = 0.5 * out + 0.5 * self.ffn2(out)
        out = self.layer_norm(out)
        return out


class _ConformerEncoder(nn.Module):
    def __init__(
        self,
        idim: int,
        n_layers: int,
        n_head: int,
        d_model: int,
        residual_dropout: float = 0.1,
        dropout_rate: float = 0.1,
        kernel_size: int = 33,
        pe_maxlen: int = 5000,
    ):
        super().__init__()
        self.odim = d_model
        self.input_preprocessor = _Conv2dSubsampling(idim, d_model)
        self.positional_encoding = _RelPositionalEncoding(d_model, max_len=pe_maxlen)
        self.dropout = nn.Dropout(residual_dropout)
        self.layer_stack = nn.ModuleList(
            [_RelPosEmbConformerBlock(d_model, n_head, residual_dropout, dropout_rate, kernel_size) for _ in range(n_layers)]
        )

    def forward(self, padded_input: torch.Tensor, input_lengths: torch.Tensor, pad: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if pad:
            padded_input = F.pad(padded_input, (0, 0, 0, self.input_preprocessor.context - 1), "constant", 0.0)
        src_mask = self._padding_position_is_0(padded_input, input_lengths)
        embed_output, input_lengths, src_mask = self.input_preprocessor(padded_input, src_mask)
        enc_output = self.dropout(embed_output)
        pos_emb = self.dropout(self.positional_encoding(embed_output))
        for enc_layer in self.layer_stack:
            enc_output = enc_layer(enc_output, pos_emb, slf_attn_mask=src_mask, pad_mask=src_mask)
        return enc_output, input_lengths, src_mask

    @staticmethod
    def _padding_position_is_0(padded_input: torch.Tensor, input_lengths: torch.Tensor) -> torch.Tensor:
        N, T = padded_input.size()[:2]
        mask = torch.ones((N, T), device=padded_input.device)
        for i in range(N):
            mask[i, input_lengths[i]:] = 0
        return mask.unsqueeze(dim=1).to(torch.uint8)


# -------------------------------
# Multimodal inputs schema
# -------------------------------


class FireRedASRAudioInputs(TensorSchema):
    input_features: Annotated[torch.Tensor, TensorShape("b", "fi", 80)]
    input_features_mask: Annotated[torch.Tensor, TensorShape("b", "fo")]
    audio_embed_sizes: Annotated[list[int], TensorShape("b")]


class FireRedASRMultiModalProcessingInfo(BaseProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": 1}

    def get_max_audio_tokens(self):
        return 15000

    def get_max_audio_len(self):
        return 8_000_000


class FireRedASRMultiModalProcessor(BaseMultiModalProcessor[FireRedASRMultiModalProcessingInfo]):
    def _get_data_parser(self) -> MultiModalDataParser:
        return MultiModalDataParser(target_sr=16000)

    def _get_mm_fields_config(self, hf_inputs: BatchFeature, hf_processor_mm_kwargs: Mapping[str, object]) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            input_features=MultiModalFieldConfig.batched("audio"),
            audio_embed_sizes=MultiModalFieldConfig.batched("audio"),
        )

    def _get_prompt_updates(
        self,
        mm_items,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs,
    ) -> list[PromptUpdate]:
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()
        speech_token = DEFAULT_SPEECH_TOKEN
        # Ensure the tokenizer knows this token; try to add if missing
        if speech_token not in vocab:
            try:
                tokenizer.add_special_tokens({"additional_special_tokens": [speech_token]})
                vocab = tokenizer.get_vocab()
            except Exception:
                warnings.warn(f"Tokenizer missing {speech_token}; merging may fail.")
        speech_token_id = vocab.get(speech_token, tokenizer.pad_token_id)

        def get_replacement(item_idx: int):
            audios = mm_items.get_items("audio", AudioProcessorItems)
            audio = audios.get(item_idx)
            # Estimate features and downsample token count based on adapter rate
            sample_rate = hf_processor_mm_kwargs.get("sample_rate", 16000)
            downsample = hf_processor_mm_kwargs.get("downsample_rate", 2)
            fbank = _SimpleFbank(sample_rate=sample_rate, n_mels=80)
            feats = fbank(audio)
            num_tokens = int(math.ceil(feats.shape[0] / downsample))
            return [speech_token_id] * max(1, num_tokens)

        return [
            PromptReplacement(modality="audio", target=[speech_token_id], replacement=get_replacement),
        ]

    def _call_hf_processor(self, prompt: str, mm_data: Mapping[str, object], mm_kwargs: Mapping[str, object], tok_kwargs: Mapping[str, object]) -> BatchFeature:
        mm_data = dict(mm_data)
        audios = mm_data.pop("audios",[])
        sample_rate = mm_kwargs.get("sample_rate", 16000)
        downsample_rate = mm_kwargs.get("downsample_rate", 2)
        fe = _ASRFeatExtractor(sample_rate=sample_rate, n_mels=80)
        batched_feats =[]
        feature_lens =[]
        audio_embed_sizes =[]
        for a in audios:
            feat = fe._fbank(a)
            batched_feats.append(feat)
            feature_lens.append(int(feat.shape[0]))
            audio_embed_sizes.append(int(math.ceil(feat.shape[0] / downsample_rate)))
        # Pad and stack
        if batched_feats:
            max_len = max(f.shape[0] for f in batched_feats)
            padded = [F.pad(f, (0, 0, 0, max_len - f.shape[0])) for f in batched_feats]
            stacked = torch.stack(padded, dim=0)
        else:
            stacked = torch.zeros((1, 0, 80))
        if batched_feats:
            input_features_mask = (
                torch.arange(0, stacked.shape[1]).view(1, -1)
                < torch.tensor(feature_lens).view(-1, 1)
            )
        else:
            input_features_mask = torch.zeros((1, 0), dtype=torch.bool)
        return BatchFeature(
            {
                "input_features": stacked,
                "input_features_mask": input_features_mask,
                "audio_embed_sizes": torch.tensor(audio_embed_sizes),
            }
        )


class FireRedASRDummyInputsBuilder:
    def get_dummy_mm_data(self, seq_len: int, mm_counts: Mapping[str, int], mm_options: Mapping[str, object] | None = None) -> dict[str, object]:
        num_audios = mm_counts.get("audio", 0)
        # Generate white noise audios
        sr = 16000
        length = sr * 2
        return {"audio": [np.random.randn(length).astype(np.float32) for _ in range(num_audios)]}

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)
        return DEFAULT_SPEECH_TOKEN * num_audios


@MULTIMODAL_REGISTRY.register_processor(
    FireRedASRMultiModalProcessor,
    info=FireRedASRMultiModalProcessingInfo,
    dummy_inputs=FireRedASRDummyInputsBuilder,
)
class FireRedASRLLMForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsPP):
    merge_by_field_config = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        cache_config = vllm_config.cache_config

        self.config: PretrainedConfig = config
        self.quant_config: QuantizationConfig | None = quant_config
        self.cache_config: CacheConfig = cache_config

        # Underlying language model (e.g., Qwen2)
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=getattr(config, "text_config", config),
            prefix=maybe_prefix(prefix, "language_model"),
        )

        # Audio encoder
        enc_conf = getattr(config, "encoder_config", None)
        idim = getattr(enc_conf, "idim", 80)
        n_layers = getattr(enc_conf, "n_layers", getattr(config, "n_layers_enc", 12))
        n_head = getattr(enc_conf, "n_head", getattr(config, "n_head", 8))
        d_model = getattr(enc_conf, "d_model", getattr(config, "d_model", 512))
        residual_dropout = getattr(enc_conf, "residual_dropout", getattr(config, "residual_dropout", 0.1))
        dropout_rate = getattr(enc_conf, "dropout_rate", getattr(config, "dropout_rate", 0.1))
        kernel_size = getattr(enc_conf, "kernel_size", getattr(config, "kernel_size", 33))
        pe_maxlen = getattr(enc_conf, "pe_maxlen", getattr(config, "pe_maxlen", 5000))

        self.encoder = _ConformerEncoder(
            idim=idim,
            n_layers=n_layers,
            n_head=n_head,
            d_model=d_model,
            residual_dropout=residual_dropout,
            dropout_rate=dropout_rate,
            kernel_size=kernel_size,
            pe_maxlen=pe_maxlen,
        )

        # Projector (Adapter)
        llm_hidden = getattr(getattr(config, "text_config", config), "hidden_size", 4096)
        downsample_rate = getattr(config, "encoder_downsample_rate", 2)
        self.projector = _Adapter(encoder_dim=self.encoder.odim, llm_dim=llm_hidden, downsample_rate=downsample_rate)

        # Try to load encoder/projector weights from local checkpoints
        self._maybe_load_local_encoder_weights()

        # Convenience alias for engine integration
        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors

    def _maybe_load_local_encoder_weights(self) -> None:
        enc_path = getattr(self.config, "asr_encoder_path", getattr(self.config, "encoder_path", None))
        if isinstance(enc_path, str):
            try:
                package = torch.load(enc_path, map_location="cpu")
                sd = package.get("model_state_dict", package)
                # Attempt to find keys for encoder
                enc_loader = AutoWeightsLoader(self.encoder)
                loaded = list(enc_loader.load_weights(sd.items()))
                if len(loaded) == 0:
                    warnings.warn("No matching weights loaded into FireRed encoder; check checkpoint format.")
            except Exception as e:
                warnings.warn(f"Failed to load encoder weights from {enc_path}: {e}")

        proj_path = getattr(self.config, "firered_model_path", getattr(self.config, "model_path", None))
        if isinstance(proj_path, str):
            try:
                package = torch.load(proj_path, map_location="cpu")
                sd = package.get("model_state_dict", package)
                proj_loader = AutoWeightsLoader(self.projector)
                loaded = list(proj_loader.load_weights(sd.items()))
                if len(loaded) == 0:
                    # Not fatal; adapter may be randomly initialized
                    pass
            except Exception:
                pass

    def get_language_model(self) -> nn.Module:
        return self.language_model

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("audio"):
            return DEFAULT_SPEECH_TOKEN
        raise ValueError("Only audio modality is supported")

    def _parse_and_validate_audio_input(self, **kwargs: object) -> FireRedASRAudioInputs | None:
        input_features = kwargs.pop("input_features", None)
        input_features_mask = kwargs.pop("input_features_mask", None)
        audio_embed_sizes = kwargs.pop("audio_embed_sizes", None)
        if input_features is None:
            return None
        if isinstance(input_features, torch.Tensor):
            # Ensure shape [bsz, num_features, 80]
            if input_features.ndim == 4:
                input_features = input_features.squeeze(1)
            if input_features.ndim != 3:
                raise ValueError("input_features must be 3D [bsz, num_features, 80]")
        else:
            raise ValueError("input_features must be a torch.Tensor")
        if input_features_mask is None:
            # Build mask from sizes
            most = input_features.shape[1]
            mask_indices = torch.arange(most, device=input_features.device).view(1, -1)
            input_features_mask = mask_indices < torch.tensor(audio_embed_sizes, device=input_features.device).view(-1, 1)
        return FireRedASRAudioInputs(
            input_features=input_features,
            input_features_mask=input_features_mask,
            audio_embed_sizes=(audio_embed_sizes.flatten().tolist() if isinstance(audio_embed_sizes, torch.Tensor) else list(audio_embed_sizes)),
        )

    def _process_audio_input(self, audio_input: FireRedASRAudioInputs) -> tuple[torch.Tensor, ...]:
        # Conformer encoder -> adapter downsample -> split by per-sample token len
        enc_lens = audio_input["input_features_mask"].sum(-1).to(torch.long)
        encoder_embeds, enc_lens, _ = self.encoder(
            audio_input["input_features"], enc_lens
        )
        projected_embeds, new_lens = self.projector(encoder_embeds, enc_lens)
        pieces =[]
        for i in range(projected_embeds.shape[0]):
            li = int(new_lens[i].item())
            pieces.append(projected_embeds[i, :li, :])
        return tuple(pieces)

    def get_multimodal_embeddings(self, **kwargs: object):
        audio_input = self._parse_and_validate_audio_input(**kwargs)
        if audio_input is None:
            return[]
        return self._process_audio_input(audio_input)

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: list[torch.Tensor] | torch.Tensor | tuple[torch.Tensor, ...] | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
        handle_oov_mm_token: bool = True,
    ) -> torch.Tensor:
        if multimodal_embeddings is None or is_multimodal is None:
            return SupportsMultiModal.get_input_embeddings(self, input_ids)
        return SupportsMultiModal.get_input_embeddings(
            self,
            input_ids,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
            handle_oov_mm_token=handle_oov_mm_token,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        # Pass through underlying language model
        return self.language_model.forward(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **{k: v for k, v in kwargs.items() if k == "token_type_ids"},
        )