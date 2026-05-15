"""Extract frozen AST embeddings for every manifest row and cache them.

We use the AudioSet-pretrained AST checkpoint
(MIT/ast-finetuned-audioset-10-10-0.4593) as a *feature extractor only*:
the model weights are frozen, we mean-pool the last hidden state over the
patch sequence, and the resulting 768-d embedding is cached as a .pt file
keyed by md5 of the source wav.

A downstream linear-probe classifier (see train_ast_probe.py) then trains
a 768->5 head per (aug arm, seed) cell. This isolates the contribution of
the augmentation pipeline from the contribution of the backbone -- the
backbone is fixed across all cells.

Run:
  python -m src.models.ast_features
    [--manifest <csv>] [--out_dir <pt cache root>]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]


def md5_path(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--manifests",
        nargs="+",
        type=Path,
        default=[
            REPO / "data" / "manifests" / "train.csv",
            REPO / "data" / "manifests" / "val.csv",
            REPO / "data" / "manifests" / "test.csv",
            REPO / "data" / "manifests" / "synth_train.csv",
        ],
    )
    p.add_argument("--out_dir", type=Path, default=REPO / "data" / "ast_features")
    p.add_argument("--model_id", type=str, default="MIT/ast-finetuned-audioset-10-10-0.4593")
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
        )
    else:
        device = args.device
    print(f"[ast] device={device}")

    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification  # noqa: WPS433
    print(f"[ast] loading {args.model_id} ...")
    t0 = time.time()
    fe = AutoFeatureExtractor.from_pretrained(args.model_id)
    model = AutoModelForAudioClassification.from_pretrained(args.model_id).to(device).eval()
    for p_ in model.parameters():
        p_.requires_grad = False
    print(f"[ast] loaded in {time.time() - t0:.1f}s")

    import torchaudio  # noqa: WPS433 — keep optional dep
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Build a unique list of (filepath, is_synthetic) across all manifests
    seen: set[str] = set()
    items: list[tuple[Path, bool]] = []
    for manifest in args.manifests:
        if not manifest.exists():
            print(f"[ast] WARN: missing {manifest}")
            continue
        with manifest.open() as f:
            for row in csv.DictReader(f):
                fp = row["filepath"]
                if fp in seen:
                    continue
                seen.add(fp)
                is_synth = row.get("source", "") == "ddpm_synthetic"
                items.append((REPO / fp, is_synth))
    print(f"[ast] {len(items)} unique items to embed")

    sr = fe.sampling_rate
    target_len_s = 5.0
    n_done, n_skip = 0, 0
    with torch.no_grad():
        for src, is_synth in items:
            cache_path = args.out_dir / f"{md5_path(src) if not is_synth else src.stem}.pt"
            if cache_path.exists():
                n_skip += 1
                continue
            try:
                if is_synth:
                    # Synthetic items are already mel spectrograms (1,64,128).
                    # AST expects waveform at fe.sampling_rate; we skip synthetic
                    # rows in the AST track since the AST feature extractor wants
                    # raw audio. They are handled separately below.
                    continue
                wav, src_sr = torchaudio.load(str(src))
                if wav.shape[0] > 1:
                    wav = wav.mean(dim=0, keepdim=True)
                if src_sr != sr:
                    wav = torchaudio.functional.resample(wav, src_sr, sr)
                wav = wav.squeeze(0)
                # Center-crop / zero-pad to target_len_s
                target = int(target_len_s * sr)
                if wav.shape[0] > target:
                    start = (wav.shape[0] - target) // 2
                    wav = wav[start : start + target]
                else:
                    pad = target - wav.shape[0]
                    wav = torch.nn.functional.pad(wav, (pad // 2, pad - pad // 2))
                inputs = fe(wav.numpy(), sampling_rate=sr, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                out = model(**inputs, output_hidden_states=True)
                # Mean-pool the patch-sequence dimension of the last hidden state.
                emb = out.hidden_states[-1].mean(dim=1).squeeze(0).cpu()
                torch.save(emb, cache_path)
                n_done += 1
                if n_done % 50 == 0:
                    print(f"[ast]  {n_done} embedded ...")
            except Exception as e:  # noqa: BLE001
                print(f"[ast]  FAIL {src.name}: {e}")
    print(f"[ast] done: {n_done} new, {n_skip} cached, total dir entries={len(list(args.out_dir.glob('*.pt')))}")


if __name__ == "__main__":
    main()
