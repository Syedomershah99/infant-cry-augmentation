"""PyTorch Dataset that reads from a manifest CSV and serves log-mel spectrograms.

The manifest schema is documented in data/README.md. Splits are fixed before
any model is trained, so a Dataset never reshuffles or merges them; the leakage
test in tests/test_no_leakage.py is the safety net.

Synthetic samples (from the diffusion model) are mixed in via a separate
manifest (`synth_train.csv`) referenced by `extra_manifests`. The classifier
treats real and synthetic rows uniformly so the only knob between aug arms is
which manifests are loaded.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import torch
from torch.utils.data import Dataset

from .features import FeatureConfig, LogMelExtractor, fix_duration, load_and_resample, standardize

LABELS = ["belly_pain", "burping", "discomfort", "hungry", "tired"]
LABEL_TO_IDX = {l: i for i, l in enumerate(LABELS)}
IDX_TO_LABEL = {i: l for l, i in LABEL_TO_IDX.items()}


@dataclass
class CryItem:
    filepath: str
    label: int
    label_str: str
    source: str
    is_synthetic: bool
    extras: dict


def read_manifest(path: Path, is_synthetic: bool = False) -> list[CryItem]:
    rows: list[CryItem] = []
    with path.open() as f:
        for row in csv.DictReader(f):
            label_str = row["label"]
            if label_str not in LABEL_TO_IDX:
                continue
            rows.append(
                CryItem(
                    filepath=row["filepath"],
                    label=LABEL_TO_IDX[label_str],
                    label_str=label_str,
                    source=row.get("source", ""),
                    is_synthetic=is_synthetic,
                    extras=row,
                )
            )
    return rows


class CryDataset(Dataset):
    """Reads raw audio per item and returns standardized log-mel.

    For synthetic rows, `filepath` is expected to point at a saved `.pt` file
    holding a precomputed (1, n_mels, frames) tensor; we skip the wav pipeline
    in that case.
    """

    def __init__(
        self,
        manifest: Path | str,
        repo_root: Path | str,
        feature_cfg: FeatureConfig | None = None,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        extra_manifests: Iterable[Path | str] = (),
    ):
        self.repo_root = Path(repo_root)
        self.cfg = feature_cfg or FeatureConfig()
        self.extractor = LogMelExtractor(self.cfg)
        self.transform = transform
        items = read_manifest(Path(manifest), is_synthetic=False)
        for em in extra_manifests:
            items.extend(read_manifest(Path(em), is_synthetic=True))
        if not items:
            raise ValueError(f"No items in {manifest}")
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def _load_real(self, item: CryItem) -> torch.Tensor:
        path = self.repo_root / item.filepath
        wav = load_and_resample(str(path), self.cfg.sample_rate)
        wav = fix_duration(wav, self.cfg.sample_rate, self.cfg.target_seconds)
        spec = self.extractor(wav.unsqueeze(0)).squeeze(0)  # (1, n_mels, frames)
        return standardize(spec)

    def _load_synth(self, item: CryItem) -> torch.Tensor:
        path = self.repo_root / item.filepath
        spec = torch.load(path, map_location="cpu")
        if spec.dim() == 2:
            spec = spec.unsqueeze(0)
        return spec  # assumed already standardized at synthesis time

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        spec = self._load_synth(item) if item.is_synthetic else self._load_real(item)
        if self.transform is not None:
            spec = self.transform(spec)
        return {
            "spec": spec.float(),
            "label": item.label,
            "label_str": item.label_str,
            "is_synthetic": int(item.is_synthetic),
            "source": item.source,
            "filepath": item.filepath,
        }


def class_counts(items: list[CryItem]) -> dict[str, int]:
    out = {l: 0 for l in LABELS}
    for it in items:
        out[it.label_str] = out.get(it.label_str, 0) + 1
    return out


def class_weights(items: list[CryItem], scheme: str = "inverse") -> torch.Tensor:
    """Class weights for weighted CE loss. `scheme`: inverse | inverse_sqrt | none."""
    counts = class_counts(items)
    raw = torch.tensor([counts[l] for l in LABELS], dtype=torch.float32).clamp_min(1.0)
    if scheme == "none":
        return torch.ones_like(raw)
    if scheme == "inverse":
        w = 1.0 / raw
    elif scheme == "inverse_sqrt":
        w = 1.0 / raw.sqrt()
    else:
        raise ValueError(scheme)
    return w * (len(LABELS) / w.sum())
