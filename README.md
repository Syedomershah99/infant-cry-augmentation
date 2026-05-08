# Infant Cry Classification with Synthetic Data Augmentation

> Final project for **CSE 4/555 — Pattern Recognition** (UB, Spring 2026).
> Reproducible study of whether **class-conditional generative augmentation** improves rare-class recall in infant cry classification, under clean and noisy/out-of-distribution conditions.

**Status:** 🚧 Phase 0 (setup). See [`PLAN.md`](#plan) for the 4-week roadmap.

---

## Why this matters

Infant cry classification supports NICU monitoring, pediatric triage, and parent-support apps. The hardest classes are also the most consequential — *pain* and *sick* are rare in public corpora yet a missed pain cry is the failure mode that matters most. This project asks a falsifiable question: **does class-conditional generative augmentation help the rare classes that classical augmentation cannot?**

## Research question

> Does class-conditional generative augmentation improve macro-F1 and rare-class recall over (a) no augmentation and (b) classical SpecAugment/pitch-shift augmentation, when training data is severely class-imbalanced — and does the gain hold under noisy / out-of-distribution test conditions?

Three controlled axes:
1. **Augmentation type:** none / classical / generative / classical+generative
2. **Synthetic-to-real ratio (rare classes only):** 0× / 1× / 5× / 10×
3. **Test condition:** clean / additive babble noise / cross-source held-out

Three random seeds per cell. Primary metrics: macro-F1, per-class recall (especially `belly_pain`), Expected Calibration Error.

## Approach

| Component | Method |
|---|---|
| Features | log-mel spectrograms (64 mels × 128 frames) |
| Baseline | ResNet-style CNN on log-mel; optionally fine-tuned AST / PANNs CNN14 |
| Generative model | Class-conditional DDPM in mel space (lightweight, ≤30M params) |
| Eval harness | per-class F1, FNR on rare classes, ECE, robustness gap, paired-bootstrap CIs |

## Datasets

- **Primary:** [donateacry-corpus](https://github.com/gveres/donateacry-corpus) — 5 classes (belly_pain, burping, discomfort, hungry, tired), heavily imbalanced. Public, parent-uploaded.
- **Cross-source eval:** public Kaggle / Zenodo infant-cry compilations (CC-licensed only).
- **Background noise:** MUSAN babble / ESC-50 ambient (CC-BY).

⚠️ **No raw audio is committed to this repo.** Manifests (filename, label, source, split) are committed; clips are downloaded via scripts and stored under a gitignored `data/raw/`.

## Ethics

This is a clinical-adjacent domain. Treat the system as **decision-support, not diagnostic**. The repo includes:

- [`ethics/DATASHEET.md`](ethics/DATASHEET.md) — dataset datasheet (Gebru et al.)
- [`ethics/MODEL_CARD.md`](ethics/MODEL_CARD.md) — model card (Mitchell et al.)
- [`ethics/bias_audit.md`](ethics/bias_audit.md) — class & demographic skew analysis
- [`ethics/consent_provenance.md`](ethics/consent_provenance.md) — per-source provenance & consent basis
- [`reading_hw/ethics_essay.pdf`](reading_hw/) — individual ethics review (course deliverable)

## Repo layout

```
configs/        # YAML configs per experiment cell
data/
  manifests/    # CSV: filepath, label, source, split, seed
  raw/          # gitignored — fetched via scripts
src/
  audio/        # log-mel, SpecAug, noise injection
  models/       # classifier + diffusion
  training/     # train_classifier, train_diffusion, sample_diffusion
  eval/         # metrics, fairness slices, robustness
  aws/          # SageMaker launch helpers
experiments/    # the experiment matrix + notebooks
ethics/         # datasheet, model card, bias audit, consent
report/         # NIPS-2017 LaTeX template (final write-up)
reading_hw/     # individual ethics essay (PDF deliverable)
tests/          # unit tests + leakage assertions
```

## Reproduction

```bash
# 1. Environment (Python 3.9+; tested on 3.9 with torch 2.8 on Apple MPS)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Data — clones donateacry-corpus into data/raw/ (gitignored), then builds manifests
mkdir -p data/raw && cd data/raw \
  && git clone --depth 1 https://github.com/gveres/donateacry-corpus.git && cd ../..
python -m src.data.build_manifests
pytest tests/test_no_leakage.py            # sanity check: no test files in train

# 3. Baseline classifier (no augmentation; intentionally collapses on rare classes)
python -m src.training.train_classifier --config configs/baseline_cnn.yaml
python -m src.training.train_classifier --config configs/baseline_classical.yaml

# 4. Conditional diffusion + sample synthetic spectrograms for rare classes
python -m src.training.train_diffusion --config configs/cond_ddpm.yaml
python -m src.training.sample_diffusion \
    --ckpt results/ddpm_seed0/best.pt \
    --per_class belly_pain=110,burping=60 \
    --cfg_scale 2.0 --steps 50

# 5. Generative augmentation arms
python -m src.training.train_classifier --config configs/baseline_generative.yaml
python -m src.training.train_classifier --config configs/baseline_classical_generative.yaml

# 6. Full experiment matrix (multi-seed, ratio sweep)
python -m experiments.run_matrix --out results/matrix.csv \
    --seeds 0 1 2 --ratios 0 1 5 10 --epochs 30
```

### Headline numbers (v1.0: 4 arms × 3 recipes × 3 seeds = 36 cells, synth ratio 3×, 30 epochs)

| Arm | Recipe | macro-F1 | accuracy | belly_pain rec. | **burping rec.** | ECE |
|---|---|---:|---:|---:|---:|---:|
| none | weighted_ce | 0.194 | 0.797 | 0.000 | 0.000 | 0.285 |
| none | balanced | 0.191 | 0.507 | **0.111** | 0.000 | **0.201** |
| none | focal | 0.213 | 0.787 | 0.000 | 0.000 | 0.354 |
| classical | weighted_ce | 0.180 | 0.807 | 0.000 | 0.000 | 0.418 |
| classical | balanced | 0.205 | 0.638 | 0.000 | 0.000 | 0.285 |
| classical | focal | 0.187 | 0.778 | 0.000 | 0.000 | 0.345 |
| **generative** | **weighted_ce** | **0.258** | **0.816** | 0.000 | **0.667** | 0.352 |
| generative | balanced | 0.195 | 0.585 | 0.000 | 0.333 | 0.226 |
| generative | focal | 0.238 | 0.763 | 0.000 | 0.333 | 0.333 |
| classical+generative | weighted_ce | 0.208 | 0.792 | 0.000 | 0.333 | 0.286 |
| classical+generative | balanced | 0.171 | 0.507 | 0.000 | 0.000 | 0.207 |
| classical+generative | focal | 0.182 | **0.836** | 0.000 | 0.000 | 0.371 |

**Reading.**
- **Generative + weighted_ce is the best cell** simultaneously on macro-F1
  (+33% over baseline) and burping recall (0.667, vs. 0.000 across all 18
  non-generative cells), without sacrificing accuracy.
- **Generative augmentation is the only intervention that recovers burping**
  — three of four generative cells produce non-zero burping recall;
  no non-generative cell does.
- **belly_pain remains hard at this data scale** (only 11 real training
  clips after kaggle dedup). The DDPM produces class-discriminative samples
  (probe recall 0.93) but the downstream classifier doesn't cross argmax.
- **The role of each axis decomposes cleanly:** larger train pool unblocks
  diffusion conditioning quality (probe burping recall 0.00 → 0.08); the
  optimizer recipe controls recall-vs-accuracy trade-off; generative
  augmentation flips the argmax for burping.

See `report/Report.pdf` for the full write-up and `results/matrix_summary.csv`
for per-cell mean+std numbers.

## Plan

Detailed 4-week plan: [`PLAN.md`](PLAN.md). Milestone tags:

- `v0.1-baseline` — log-mel + CNN baseline reproduced, eval harness done
- `v0.2-diffusion` — class-conditional DDPM trained, sample-quality checks pass
- `v0.3-results` — full experiment matrix run, fairness audit complete
- `v1.0-submission` — report compiled, model card + datasheet finalized

## License

MIT. See [`LICENSE`](LICENSE).

## Citation

If you use this work, please cite (placeholder, will update on submission):

```bibtex
@misc{shah2026infantcry,
  title  = {Class-Conditional Generative Augmentation for Rare-Class Infant Cry Classification},
  author = {Shah, Syed Omer},
  year   = {2026},
  note   = {CSE 4/555 Final Project, University at Buffalo}
}
```
