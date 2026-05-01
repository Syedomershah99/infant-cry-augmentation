"""Pytest assertion: test-split files never appear in synthesis or augmentation manifests.

Skipped when manifests are not yet built. Will become hard-required at v0.1-baseline.
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MANIFEST_DIR = REPO / "data" / "manifests"


def _maybe_load(name: str):
    p = MANIFEST_DIR / name
    if not p.exists():
        pytest.skip(f"{p} not built yet")
    import csv
    with p.open() as f:
        return list(csv.DictReader(f))


def test_no_test_files_in_train_or_synth():
    train = _maybe_load("train.csv")
    test = _maybe_load("test.csv")
    test_files = {row["filepath"] for row in test}
    train_files = {row["filepath"] for row in train}
    assert train_files.isdisjoint(test_files), "Test files leaked into train manifest"

    synth_path = MANIFEST_DIR / "synth_train.csv"
    if synth_path.exists():
        import csv
        with synth_path.open() as f:
            synth_files = {row["source_filepath"] for row in csv.DictReader(f) if "source_filepath" in row}
        assert synth_files.isdisjoint(test_files), "Test files used as synthesis sources"


def test_no_cross_test_files_in_train():
    train = _maybe_load("train.csv")
    cross = _maybe_load("cross_test.csv")
    train_files = {row["filepath"] for row in train}
    cross_files = {row["filepath"] for row in cross}
    assert train_files.isdisjoint(cross_files), "Cross-source test files leaked into train"
