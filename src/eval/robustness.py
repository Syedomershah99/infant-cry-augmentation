"""Noise-robustness evaluation in mel space.

Loads a trained classifier checkpoint, builds the test dataloader, and evaluates
the classifier under additive Gaussian noise on the standardized log-mel
spectrograms at multiple SNR levels.

We add noise to the standardized spectrogram, so signal power is ~1 and SNR (dB)
maps to noise std as sigma = 10 ** (-snr_db/20). This gives:
  clean:   sigma = 0
  20 dB:   sigma ≈ 0.10
  10 dB:   sigma ≈ 0.316
  5 dB:    sigma ≈ 0.562
  0 dB:    sigma = 1.0

Output: a tidy CSV with one row per (checkpoint, snr_db, metric).

Run:
  python -m src.eval.robustness \
      --ckpts results/matrix/generative_weighted_ce_r3_s0/best.pt \
              results/matrix/generative_weighted_ce_r3_s1/best.pt \
              results/matrix/generative_weighted_ce_r3_s2/best.pt \
      --snrs clean 20 10 5 0 \
      --out results/robustness.csv
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.audio.dataset import CryDataset, LABELS
from src.audio.features import FeatureConfig
from src.eval.metrics import per_class_prf, expected_calibration_error
from src.models.classifier import CryCNN

REPO = Path(__file__).resolve().parents[2]


def snr_to_sigma(snr_db: float | str) -> float:
    if isinstance(snr_db, str) and snr_db.lower() == "clean":
        return 0.0
    return 10 ** (-float(snr_db) / 20.0)


def load_classifier(ckpt_path: Path, device: str) -> tuple[CryCNN, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    model = CryCNN(
        num_classes=len(LABELS),
        base_channels=cfg.get("base_channels", 32),
        dropout=cfg.get("dropout", 0.2),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def evaluate_with_noise(
    model: CryCNN,
    loader: DataLoader,
    sigma: float,
    device: str,
    seed: int = 0,
) -> dict:
    """Evaluate the classifier with N(0, sigma^2) noise added to each test spec.

    Noise is applied per-sample with a deterministic generator so the same SNR
    level produces the same noisy test set across runs of this script.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            spec = batch["spec"]
            if sigma > 0:
                noise = torch.randn(spec.shape, generator=g) * sigma
                spec = spec + noise
            spec = spec.to(device)
            logits = model(spec)
            all_logits.append(logits.cpu())
            all_labels.append(batch["label"].cpu())
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels).numpy()
    probs = torch.softmax(logits, dim=-1).numpy()
    preds = probs.argmax(axis=1)
    pc = per_class_prf(labels, preds)
    macro_f1 = float(np.mean([pc[l]["f1"] for l in LABELS]))
    return {
        "macro_f1": macro_f1,
        "accuracy": float(np.mean(labels == preds)),
        "ece": expected_calibration_error(probs, labels),
        "belly_pain_recall": pc["belly_pain"]["recall"],
        "burping_recall": pc["burping"]["recall"],
        "discomfort_recall": pc["discomfort"]["recall"],
        "hungry_recall": pc["hungry"]["recall"],
        "tired_recall": pc["tired"]["recall"],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", type=Path, nargs="+", required=True,
                   help="Checkpoint(s) to evaluate. Cell name (parent dir) is recorded per row.")
    p.add_argument("--snrs", nargs="+", default=["clean", "20", "10", "5", "0"],
                   help="SNR levels in dB; use 'clean' for no noise.")
    p.add_argument("--manifest", type=Path, default=REPO / "data" / "manifests" / "test.csv")
    p.add_argument("--out", type=Path, default=REPO / "results" / "robustness.csv")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--batch_size", type=int, default=32)
    args = p.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    print(f"[robustness] device={device}, ckpts={len(args.ckpts)}, snrs={args.snrs}")

    rows: list[dict] = []
    for ckpt_path in args.ckpts:
        cell_name = ckpt_path.parent.name
        model, cfg = load_classifier(ckpt_path, device)
        feat_cfg = FeatureConfig(**cfg.get("features", {}))
        test_ds = CryDataset(
            manifest=REPO / args.manifest,
            repo_root=REPO,
            feature_cfg=feat_cfg,
        )
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
        for snr_label in args.snrs:
            sigma = snr_to_sigma(snr_label)
            metrics = evaluate_with_noise(model, test_loader, sigma, device, seed=42)
            row = {
                "cell": cell_name,
                "snr_db": snr_label,
                "sigma": round(sigma, 4),
                **{k: round(v, 4) for k, v in metrics.items()},
            }
            rows.append(row)
            print(
                f"  {cell_name:32s}  snr={snr_label:>5}  "
                f"macroF1={row['macro_f1']:.3f}  acc={row['accuracy']:.3f}  "
                f"bp_rec={row['belly_pain_recall']:.3f}  burp_rec={row['burping_recall']:.3f}  "
                f"ece={row['ece']:.3f}"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[robustness] wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
