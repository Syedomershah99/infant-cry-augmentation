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

### Headline numbers so far (seed=0, 30 epochs)

| Augmentation | macro-F1 | belly_pain recall | burping recall | ECE |
|---|---:|---:|---:|---:|
| none | 0.183 | **0.00** | **0.00** | 0.48 |
| classical | 0.183 | **0.00** | **0.00** | 0.51 |
| generative | _(pending Phase 2)_ | | | |
| classical+generative | _(pending Phase 2)_ | | | |

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
