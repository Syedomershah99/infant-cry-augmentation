"""Build train/val/test manifests for the cry classifier.

Two corpora are integrated:

1. donateacry-corpus (primary). The val/test splits are drawn from this corpus
   only, so the held-out evaluation is on real, parent-uploaded clips.

2. Donate-A-Cry-Augmented (Kaggle). This corpus is the union of (1) plus
   programmatic augmentations of the rare classes (~107 extra clips per rare
   class; 0 extra hungry clips). We add the kaggle_only rows to the train split
   so the diffusion model and classifier see far more rare-class examples,
   without changing val/test.

This deliberate asymmetry keeps results comparable to the donateacry-only baseline
while letting the rare-class learning lean on the augmented pool.

Outputs:
  data/manifests/all.csv           every clip we know about (incl. kaggle_only)
  data/manifests/train.csv         donateacry train + kaggle_only
  data/manifests/val.csv           donateacry val (real only)
  data/manifests/test.csv          donateacry test (real only)
  data/manifests/cross_test.csv    optional: donateacry-only val+test, written when
                                   the kaggle-only rows ever leak in (currently
                                   empty placeholder for future cross-source corpus)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import random
import re
import wave
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_DON = REPO / "data" / "raw" / "donateacry-corpus" / "donateacry_corpus_cleaned_and_updated_data"
DEFAULT_KAGGLE = REPO / "data" / "raw" / "kaggle_archive" / "cry"
DEFAULT_OUT = REPO / "data" / "manifests"

LABELS = ["belly_pain", "burping", "discomfort", "hungry", "tired"]
KAGGLE_LABEL_DIR = {
    "belly_pain": "belly pain",
    "burping": "burping",
    "discomfort": "discomfort",
    "hungry": "hungry",
    "tired": "tired",
}

DON_FILENAME_RE = re.compile(
    r"^(?P<uuid>[A-Za-z0-9-]+)-(?P<ts>\d+)-(?P<v>[\d.]+)-(?P<gender>[mf])-(?P<weeks>\d+)-(?P<code>[a-z]+)\.wav$"
)
CODE_TO_LABEL = {"bp": "belly_pain", "bu": "burping", "dc": "discomfort", "hu": "hungry", "ti": "tired"}


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_donateacry_filename(name: str) -> dict:
    m = DON_FILENAME_RE.match(name)
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
    try:
        with wave.open(str(path), "rb") as w:
            return w.getnframes() / max(1, w.getframerate()), w.getframerate()
    except Exception:  # noqa: BLE001 — corpus has occasional non-PCM files
        return 0.0, 0


def collect_donateacry(raw_dir: Path) -> tuple[list[dict], dict[str, str]]:
    """Returns (rows, hash_to_relpath)."""
    rows: list[dict] = []
    by_hash: dict[str, str] = {}
    for label in LABELS:
        cls_dir = raw_dir / label
        if not cls_dir.exists():
            continue
        for wav in sorted(cls_dir.glob("*.wav")):
            duration, sr = wav_meta(wav)
            meta = parse_donateacry_filename(wav.name)
            rel = str(wav.relative_to(REPO))
            rows.append(
                {
                    "filepath": rel,
                    "label": label,
                    "source": "donateacry",
                    "duration_s": f"{duration:.3f}",
                    "sample_rate": str(sr),
                    "consent_basis": "donateacry_public_corpus",
                    "uuid": meta["uuid"],
                    "gender": meta["gender"],
                    "weeks": meta["weeks"],
                    "filename_label_mismatch": (
                        "1" if meta["filename_label"] and meta["filename_label"] != label else "0"
                    ),
                }
            )
            by_hash[md5_of(wav)] = rel
    return rows, by_hash


def collect_kaggle_extras(raw_dir: Path, donateacry_hashes: set[str]) -> list[dict]:
    """Return only the rows whose md5 is NOT already in donateacry."""
    rows: list[dict] = []
    for label in LABELS:
        kdir = raw_dir / KAGGLE_LABEL_DIR[label]
        if not kdir.exists():
            continue
        for wav in sorted(kdir.glob("*.wav")):
            h = md5_of(wav)
            if h in donateacry_hashes:
                continue
            duration, sr = wav_meta(wav)
            rel = str(wav.relative_to(REPO))
            rows.append(
                {
                    "filepath": rel,
                    "label": label,
                    "source": "kaggle_augmented",
                    "duration_s": f"{duration:.3f}",
                    "sample_rate": str(sr),
                    "consent_basis": "kaggle_donateacry_augmented",
                    "uuid": "",
                    "gender": "",
                    "weeks": "",
                    "filename_label_mismatch": "0",
                }
            )
    return rows


def stratified_split(
    rows: list[dict],
    seed: int = 42,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> dict[str, list[dict]]:
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
    "filepath", "label", "source", "split", "duration_s", "sample_rate",
    "seed_for_split", "consent_basis", "uuid", "gender", "weeks",
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
    out = {l: 0 for l in LABELS}
    for r in rows:
        out[r["label"]] = out.get(r["label"], 0) + 1
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--donateacry", type=Path, default=DEFAULT_DON)
    p.add_argument("--kaggle", type=Path, default=DEFAULT_KAGGLE)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    don_rows, don_hashes = collect_donateacry(args.donateacry)
    if not don_rows:
        raise SystemExit(f"No donateacry rows found under {args.donateacry}")
    print(f"[manifest] donateacry rows: {len(don_rows)}, by class: {class_counts(don_rows)}")

    splits = stratified_split(don_rows, seed=args.seed)
    print(
        f"[manifest] donateacry split (seed={args.seed}): "
        f"train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}"
    )

    kaggle_rows: list[dict] = []
    if args.kaggle.exists():
        kaggle_rows = collect_kaggle_extras(args.kaggle, set(don_hashes.keys()))
        print(f"[manifest] kaggle extras (not in donateacry): {len(kaggle_rows)}, by class: {class_counts(kaggle_rows)}")
    else:
        print(f"[manifest] kaggle dir {args.kaggle} not present; skipping")

    # Add all kaggle extras to the train split.
    train_rows = list(splits["train"])
    for r in kaggle_rows:
        r = dict(r, split="train", seed_for_split=str(args.seed))
        train_rows.append(r)

    val_rows = splits["val"]
    test_rows = splits["test"]

    write_csv(args.out / "all.csv", [dict(r, split="all", seed_for_split=str(args.seed)) for r in (don_rows + kaggle_rows)])
    write_csv(args.out / "train.csv", train_rows)
    write_csv(args.out / "val.csv", val_rows)
    write_csv(args.out / "test.csv", test_rows)
    # cross_test placeholder: empty until a true cross-source corpus is integrated.
    write_csv(args.out / "cross_test.csv", [])

    print(f"[manifest] wrote manifests to {args.out}")
    print(f"[manifest] FINAL train: n={len(train_rows)}  classes={class_counts(train_rows)}")
    print(f"[manifest] FINAL val:   n={len(val_rows)}    classes={class_counts(val_rows)}")
    print(f"[manifest] FINAL test:  n={len(test_rows)}   classes={class_counts(test_rows)}")


if __name__ == "__main__":
    main()
