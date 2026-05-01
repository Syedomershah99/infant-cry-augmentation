"""Log-mel spectrogram extraction for the cry classifier.

The whole pipeline operates in log-mel space: classifier input, diffusion training,
and synthesis. This module is the single source of truth for feature shape and
normalization, so changes here propagate to both the classifier and the diffusion
model and stay consistent across train/val/test.

Defaults: 16 kHz mono, 5 s clips, 64 mel bands, n_fft=1024, hop=512.
At 16 kHz with hop=512 a 5 s clip produces ceil(80000/512) = 157 frames; we crop
or pad to TARGET_FRAMES=128 so all clips share a single (1, 64, 128) tensor shape.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torchaudio


@dataclass(frozen=True)
class FeatureConfig:
    sample_rate: int = 16_000
    n_mels: int = 64
    n_fft: int = 1024
    hop_length: int = 512
    f_min: float = 50.0
    f_max: float = 8_000.0
    target_frames: int = 128
    target_seconds: float = 5.0
    log_offset: float = 1e-6
    # Per-feature standardization is applied at the dataset level after computing
    # log-mel; the classifier and diffusion models both expect ~zero-mean inputs.


def load_and_resample(path: str, sample_rate: int) -> torch.Tensor:
    """Load a wav, mix to mono, resample to target rate. Returns (1, T)."""
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    return waveform


def fix_duration(waveform: torch.Tensor, sample_rate: int, target_seconds: float) -> torch.Tensor:
    """Pad with zeros or center-crop to a fixed duration. Input/output (1, T)."""
    target = int(round(target_seconds * sample_rate))
    t = waveform.shape[-1]
    if t == target:
        return waveform
    if t > target:
        start = (t - target) // 2
        return waveform[..., start : start + target]
    pad = target - t
    left = pad // 2
    right = pad - left
    return torch.nn.functional.pad(waveform, (left, right))


def fix_frames(spec: torch.Tensor, target_frames: int) -> torch.Tensor:
    """Pad or center-crop along the time axis (last dim) to target_frames."""
    t = spec.shape[-1]
    if t == target_frames:
        return spec
    if t > target_frames:
        start = (t - target_frames) // 2
        return spec[..., start : start + target_frames]
    pad = target_frames - t
    left = pad // 2
    right = pad - left
    return torch.nn.functional.pad(spec, (left, right))


class LogMelExtractor(torch.nn.Module):
    """Waveform (B, 1, T) or (1, T) -> log-mel (B, 1, n_mels, target_frames).

    The leading channel dim is preserved so the classifier can treat log-mel
    spectrograms as 1-channel images and the diffusion model can do the same.
    """

    def __init__(self, cfg: FeatureConfig | None = None):
        super().__init__()
        self.cfg = cfg or FeatureConfig()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.cfg.sample_rate,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            n_mels=self.cfg.n_mels,
            f_min=self.cfg.f_min,
            f_max=self.cfg.f_max,
            power=2.0,
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)  # (1, 1, T)
        if waveform.dim() == 3 and waveform.shape[1] != 1:
            waveform = waveform.mean(dim=1, keepdim=True)
        mel = self.mel(waveform.squeeze(1))  # (B, n_mels, T')
        log_mel = torch.log(mel + self.cfg.log_offset)
        log_mel = fix_frames(log_mel, self.cfg.target_frames)
        return log_mel.unsqueeze(1)  # (B, 1, n_mels, target_frames)


def standardize(spec: torch.Tensor) -> torch.Tensor:
    """Per-spectrogram standardization. Stable across recording-channel variation."""
    mean = spec.mean(dim=(-2, -1), keepdim=True)
    std = spec.std(dim=(-2, -1), keepdim=True).clamp_min(1e-5)
    return (spec - mean) / std


def waveform_to_logmel(path: str, cfg: FeatureConfig | None = None) -> torch.Tensor:
    """Convenience for offline feature inspection: wav path -> (1, n_mels, frames)."""
    cfg = cfg or FeatureConfig()
    wav = load_and_resample(path, cfg.sample_rate)
    wav = fix_duration(wav, cfg.sample_rate, cfg.target_seconds)
    extractor = LogMelExtractor(cfg)
    spec = extractor(wav.unsqueeze(0))  # (1, 1, n_mels, frames)
    return standardize(spec).squeeze(0)
