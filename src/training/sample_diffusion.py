"""Sample synthetic spectrograms from a trained class-conditional DDPM.

Writes each sample to data/synth/<class>/<seed>_<idx>.pt as a (1, 64, 128) tensor
and appends a row to data/manifests/synth_train.csv. The classifier dataset
honours the `is_synthetic` flag and loads the .pt directly, skipping the WAV
pipeline.

Run (typical):
  python -m src.training.sample_diffusion \
      --ckpt results/ddpm_seed0/best.pt \
      --per_class belly_pain=110,burping=60 \
      --out data/synth \
      --manifest data/manifests/synth_train.csv \
      --cfg_scale 2.0 --steps 50
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from src.audio.dataset import LABEL_TO_IDX, LABELS
from src.models.diffusion import CondUNet, DDPM, DDPMConfig

REPO = Path(__file__).resolve().parents[2]


def parse_per_class(spec: str) -> dict[str, int]:
    """e.g. 'belly_pain=110,burping=60' -> {'belly_pain': 110, 'burping': 60}"""
    out: dict[str, int] = {}
    for piece in spec.split(","):
        k, v = piece.split("=")
        k = k.strip()
        if k not in LABEL_TO_IDX:
            raise ValueError(f"Unknown class: {k}")
        out[k] = int(v)
    return out


def load_ddpm(ckpt_path: Path, device: str) -> DDPM:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["cfg"]
    unet = CondUNet(
        num_classes=len(LABELS),
        base_ch=cfg.get("base_ch", 64),
        ch_mults=tuple(cfg.get("ch_mults", [1, 2, 4])),
        dropout=cfg.get("dropout", 0.1),
    )
    ddpm = DDPM(unet, DDPMConfig(**cfg.get("ddpm", {}))).to(device)
    ddpm.load_state_dict(ckpt["model"])
    ddpm.eval()
    return ddpm


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--per_class", type=str, required=True, help="e.g. belly_pain=110,burping=60")
    p.add_argument("--out", type=Path, default=REPO / "data" / "synth")
    p.add_argument("--manifest", type=Path, default=REPO / "data" / "manifests" / "synth_train.csv")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--cfg_scale", type=float, default=2.0)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--n_mels", type=int, default=64)
    p.add_argument("--frames", type=int, default=128)
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
    print(f"[device] {device}")

    torch.manual_seed(args.seed)
    counts = parse_per_class(args.per_class)
    print(f"[plan] generating {counts}")

    ddpm = load_ddpm(args.ckpt, device=device)
    args.out.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    new_rows: list[dict] = []
    for label, n in counts.items():
        cls_dir = args.out / label
        cls_dir.mkdir(parents=True, exist_ok=True)
        cls_idx = LABEL_TO_IDX[label]
        i = 0
        while i < n:
            this_b = min(args.batch, n - i)
            y = torch.full((this_b,), cls_idx, device=device, dtype=torch.long)
            x = ddpm.ddim_sample(
                shape=(this_b, 1, args.n_mels, args.frames),
                y=y,
                steps=args.steps,
                cfg_scale=args.cfg_scale,
                device=device,
            ).cpu()
            for k in range(this_b):
                fname = f"{args.seed}_{i + k:05d}.pt"
                torch.save(x[k], cls_dir / fname)
                new_rows.append(
                    {
                        "filepath": str((cls_dir / fname).relative_to(REPO)),
                        "label": label,
                        "source": "ddpm_synthetic",
                        "split": "train",
                        "duration_s": "",
                        "sample_rate": "",
                        "seed_for_split": "",
                        "consent_basis": "synthetic_no_real_clip",
                        "uuid": "",
                        "gender": "",
                        "weeks": "",
                        "filename_label_mismatch": "0",
                        "source_filepath": "",  # synth was not derived from a single train file
                    }
                )
            i += this_b
        print(f"[sampled] {label}: {n}")

    # Append to manifest (create if not present).
    fieldnames = list(new_rows[0].keys()) if new_rows else []
    write_header = not args.manifest.exists()
    with args.manifest.open("a", newline="") as f:
        if not fieldnames:
            return
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        for row in new_rows:
            w.writerow(row)
    print(f"[manifest] wrote {len(new_rows)} rows to {args.manifest}")


if __name__ == "__main__":
    main()
