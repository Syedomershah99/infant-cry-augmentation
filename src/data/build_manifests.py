"""Build train/val/test manifests for the donateacry-corpus.

Reads WAVs under data/raw/donateacry-corpus/donateacry_corpus_cleaned_and_updated_data/<label>/,
extracts duration + sample_rate via the stdlib `wave` module (no external deps), and writes:

  data/manifests/all.csv
  data/manifests/train.csv
  data/manifests/val.csv
  data/manifests/test.csv

Splits are class-stratified, deterministic, and assigned at the file level — no clip
appears in more than one split. Filenames in donateacry encode age/gender metadata; we
parse them into separate columns to enable a bias audit but keep the raw filename
intact in `filepath` for traceability.

Run:
  python -m src.data.build_manifests
"""
from __future__ import annotations

import argparse
import csv
import random
import re
import wave
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_RAW = REPO / "data" / "raw" / "donateacry-corpus" / "donateacry_corpus_cleaned_and_updated_data"
DEFAULT_OUT = REPO / "data" / "manifests"

LABELS = ["belly_pain", "burping", "discomfort", "hungry", "tired"]
SOURCE = "donateacry"
CONSENT_BASIS = "donateacry_public_corpus"

# Filename pattern observed in the cleaned corpus. Examples:
#   999bf14b-e417-4b44-b746-9253f81efe38-1430974001343-1.7-m-26-bp.wav
#   C421C6FE-DFEE-...-1430548333-1.0-f-26-bp.wav
# We extract age (float seconds-of-clip-or-version), gender (m/f), and the trailing
# class code (bp/bu/dc/hu/ti). The middle digits are not always reliable, so we keep
# parsing best-effort and tolerate failures.
FILENAME_RE = re.compile(
    r"^(?P<uuid>[A-Za-z0-9-]+)-(?P<ts>\d+)-(?P<v>[\d.]+)-(?P<gender>[mf])-(?P<weeks>\d+)-(?P<code>[a-z]+)\.wav$"
)
CODE_TO_LABEL = {"bp": "belly_pain", "bu": "burping", "dc": "discomfort", "hu": "hungry", "ti": "tired"}


def parse_filename(name: str) -> dict:
    m = FILENAME_RE.match(name)
    if not m:
        return {"uuid": "", "gender": "", "weeks": "", "filename_label": ""}
    g = m.groupdict()
    return {
        "uuid": g["uuid"],
        "gender": g["gender"],
        "weeks": g["weeks"],
        "filename_label": CODE_TO_LABEL.get(g["code"], ""),
    }


def wav_meta(path: Path) -> tuple[float, int]:
    """Return (duration_seconds, sample_rate) using stdlib only."""
    with wave.open(str(path), "rb") as w:
        frames = w.getnframes()
        sr = w.getframerate()
    return frames / sr if sr else 0.0, sr


def build_all(raw_dir: Path) -> list[dict]:
    rows = []
    for label in LABELS:
        cls_dir = raw_dir / label
        if not cls_dir.exists():
            continue
        for wav in sorted(cls_dir.glob("*.wav")):
            try:
                duration, sr = wav_meta(wav)
            except Exception:  # noqa: BLE001 — corpus has occasional non-PCM files
                duration, sr = 0.0, 0
            meta = parse_filename(wav.name)
            # Sanity: filename label code should agree with directory label.
            if meta["filename_label"] and meta["filename_label"] != label:
                # Trust the directory; record the discrepancy in a column for the audit.
                meta["filename_label_mismatch"] = "1"
            else:
                meta["filename_label_mismatch"] = "0"
            rows.append(
                {
                    "filepath": str(wav.relative_to(REPO)),
                    "label": label,
                    "source": SOURCE,
                    "duration_s": f"{duration:.3f}",
                    "sample_rate": str(sr),
                    "consent_basis": CONSENT_BASIS,
                    "uuid": meta["uuid"],
                    "gender": meta["gender"],
                    "weeks": meta["weeks"],
                    "filename_label_mismatch": meta["filename_label_mismatch"],
                }
            )
    return rows


def stratified_split(
    rows: list[dict],
    seed: int = 42,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> dict[str, list[dict]]:
    """Class-stratified deterministic split. Test = remainder."""
    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = {l: [] for l in LABELS}
    for r in rows:
        by_label.setdefault(r["label"], []).append(r)

    train, val, test = [], [], []
    for label, items in by_label.items():
        rng.shuffle(items)
        n = len(items)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        # Guard tiny classes: ensure ≥1 in val and test if class has ≥3 clips.
        if n >= 3:
            n_val = max(1, n_val)
            n_test = max(1, n - n_train - n_val)
            n_train = n - n_val - n_test
        else:
            n_train, n_val, n_test = n, 0, 0
        for i, item in enumerate(items):
            item = dict(item, seed_for_split=str(seed))
            if i < n_train:
                item["split"] = "train"
                train.append(item)
            elif i < n_train + n_val:
                item["split"] = "val"
                val.append(item)
            else:
                item["split"] = "test"
                test.append(item)
    return {"train": train, "val": val, "test": test}


COLUMNS = [
    "filepath",
    "label",
    "source",
    "split",
    "duration_s",
    "sample_rate",
    "seed_for_split",
    "consent_basis",
    "uuid",
    "gender",
    "weeks",
    "filename_label_mismatch",
]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def class_counts(rows: list[dict]) -> dict[str, int]:
    counts = {l: 0 for l in LABELS}
    for r in rows:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    return counts


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rows = build_all(args.raw)
    if not rows:
        raise SystemExit(f"No WAVs found under {args.raw}")

    splits = stratified_split(rows, seed=args.seed)
    write_csv(args.out / "all.csv", [dict(r, split="all", seed_for_split=str(args.seed)) for r in rows])
    for name, items in splits.items():
        write_csv(args.out / f"{name}.csv", items)

    print(f"Built manifests in {args.out}")
    print(f"  total: {len(rows)} clips")
    print(f"  class counts (all): {class_counts(rows)}")
    for name, items in splits.items():
        print(f"  {name}: n={len(items)}, classes={class_counts(items)}")


if __name__ == "__main__":
    main()
