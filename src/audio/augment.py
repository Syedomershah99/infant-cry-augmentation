"""Classical augmentations applied in log-mel space.

These are the ``classical'' arm of the experiment matrix: SpecAugment-style
time/frequency masking, additive Gaussian noise, and small time/freq shifts.
They are applied on the spectrogram tensor, after standardization. The
generative augmentation arm (synthetic spectrograms from the conditional DDPM)
is composed at the dataset level; this module only handles the classical ops.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import torch


@dataclass
class SpecAugConfig:
    time_mask_param: int = 16     # max width of a single time mask (frames)
    freq_mask_param: int = 8      # max width of a single freq mask (mel bins)
    n_time_masks: int = 2
    n_freq_masks: int = 2
    noise_std: float = 0.02       # std of additive Gaussian noise (post-standardize)
    time_shift_frac: float = 0.1  # max fraction of frames to shift left/right
    apply_prob: float = 0.8       # probability of applying any augmentation at all


def time_mask(spec: torch.Tensor, max_param: int, n_masks: int, rng: random.Random) -> torch.Tensor:
    if max_param <= 0 or n_masks <= 0:
        return spec
    out = spec.clone()
    frames = out.shape[-1]
    for _ in range(n_masks):
        w = rng.randint(0, max_param)
        if w == 0:
            continue
        start = rng.randint(0, max(0, frames - w))
        out[..., :, start : start + w] = 0.0
    return out


def freq_mask(spec: torch.Tensor, max_param: int, n_masks: int, rng: random.Random) -> torch.Tensor:
    if max_param <= 0 or n_masks <= 0:
        return spec
    out = spec.clone()
    mels = out.shape[-2]
    for _ in range(n_masks):
        h = rng.randint(0, max_param)
        if h == 0:
            continue
        start = rng.randint(0, max(0, mels - h))
        out[..., start : start + h, :] = 0.0
    return out


def time_shift(spec: torch.Tensor, max_frac: float, rng: random.Random) -> torch.Tensor:
    if max_frac <= 0:
        return spec
    frames = spec.shape[-1]
    max_shift = int(round(frames * max_frac))
    if max_shift == 0:
        return spec
    shift = rng.randint(-max_shift, max_shift)
    return torch.roll(spec, shifts=shift, dims=-1)


def add_noise(spec: torch.Tensor, std: float, rng_torch: torch.Generator | None) -> torch.Tensor:
    if std <= 0:
        return spec
    return spec + torch.randn(spec.shape, generator=rng_torch) * std


class ClassicalAug:
    """Stateful, reproducible classical augmentation."""

    def __init__(self, cfg: SpecAugConfig | None = None, seed: int | None = None):
        self.cfg = cfg or SpecAugConfig()
        self._rng = random.Random(seed)
        self._torch_rng = torch.Generator()
        if seed is not None:
            self._torch_rng.manual_seed(seed)

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        if self._rng.random() > cfg.apply_prob:
            return spec
        spec = time_shift(spec, cfg.time_shift_frac, self._rng)
        spec = freq_mask(spec, cfg.freq_mask_param, cfg.n_freq_masks, self._rng)
        spec = time_mask(spec, cfg.time_mask_param, cfg.n_time_masks, self._rng)
        spec = add_noise(spec, cfg.noise_std, self._torch_rng)
        return spec
