"""Smoke tests for the feature/dataset pipeline.

Skipped automatically when the heavy deps (torch/torchaudio) are not installed,
so this can run in CI without GPU/torch.
"""
from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
torchaudio = pytest.importorskip("torchaudio")

from src.audio.dataset import CryDataset, LABELS  # noqa: E402
from src.audio.features import FeatureConfig, LogMelExtractor  # noqa: E402
from src.audio.augment import ClassicalAug, SpecAugConfig  # noqa: E402
from src.models.classifier import CryCNN  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def test_logmel_shape_from_synthetic_waveform():
    cfg = FeatureConfig()
    extractor = LogMelExtractor(cfg)
    wav = torch.randn(1, 1, cfg.sample_rate * int(cfg.target_seconds))
    spec = extractor(wav)
    assert spec.shape == (1, 1, cfg.n_mels, cfg.target_frames)


def test_classical_aug_preserves_shape():
    cfg = FeatureConfig()
    spec = torch.randn(1, cfg.n_mels, cfg.target_frames)
    aug = ClassicalAug(SpecAugConfig(), seed=0)
    out = aug(spec)
    assert out.shape == spec.shape


def test_classifier_forward_shape():
    model = CryCNN(num_classes=len(LABELS), base_channels=8)
    x = torch.randn(2, 1, 64, 128)
    logits = model(x)
    assert logits.shape == (2, len(LABELS))


@pytest.mark.skipif(
    not (REPO / "data" / "manifests" / "train.csv").exists(),
    reason="train manifest not built yet",
)
@pytest.mark.skipif(
    not (REPO / "data" / "raw" / "donateacry-corpus").exists(),
    reason="raw corpus not pulled (data/raw is gitignored)",
)
def test_dataset_loads_one_real_clip():
    ds = CryDataset(
        manifest=REPO / "data" / "manifests" / "train.csv",
        repo_root=REPO,
    )
    item = ds[0]
    assert item["spec"].shape == (1, 64, 128)
    assert 0 <= item["label"] < len(LABELS)
    assert item["is_synthetic"] == 0
