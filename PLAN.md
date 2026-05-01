# Plan

Detailed 4-week roadmap for this project. The authoritative plan lives here in the repo so collaborators and graders can see the design without context outside the repository.

## Goal

Answer one falsifiable question: **does class-conditional generative augmentation improve rare-class recall over classical augmentation in infant cry classification, and does the gain hold under noisy / out-of-distribution conditions?**

The three required deliverables (per course spec):

1. **Reproduce a baseline.** Log-mel + CNN classifier (or fine-tuned AST / PANNs CNN14) on donateacry-corpus.
2. **Add one extension.** Class-conditional DDPM on log-mel spectrograms; rare-class-targeted augmentation pipeline.
3. **Evaluate one carefully defined question.** The matrix in the [Experiment design](#experiment-design) section below.

## Timeline (May 1 → May 29, 2026)

### Phase 0 — Setup & Reading HW (May 1–5) — gated by Reading HW deadline 2026-05-05

- [x] Create public repo + skeleton (this commit)
- [ ] Email instructor: request solo approval (Risk #1)
- [ ] Pull donateacry-corpus → write `data/manifests/{train,val,test}.csv` (source-stratified splits)
- [ ] EDA notebook: class counts, durations, sample-rate, spectrogram visuals
- [ ] **Draft + submit Reading HW** (`reading_hw/ethics_essay.pdf`) — due 2026-05-05 23:59
- [ ] Log-mel feature extractor + dataloader skeleton + 1-epoch smoke test

### Phase 1 — Baseline (May 6–12)

- [ ] ResNet-style CNN baseline on log-mel; 3 seeds; per-class F1
- [ ] Classical augmentations: SpecAugment, pitch-shift, time-shift, additive Gaussian noise
- [ ] Eval harness: macro-F1, per-class P/R, confusion matrix, ECE, calibration plot
- [ ] (Optional, time permitting) AST or PANNs CNN14 fine-tune; pick stronger as canonical baseline
- [ ] Tag `v0.1-baseline`

### Phase 2 — Generative extension (May 13–20)

- [ ] Class-conditional DDPM on 64×128 log-mel patches (label embedding ⊕ time embedding)
- [ ] Train on AWS SageMaker (`ml.g5.xlarge`), checkpoints to S3
- [ ] Sample-quality checks: FAD proxy + held-out classifier consistency on synthetic
- [ ] Listening checks via Griffin-Lim / HiFi-GAN inversion (sanity only)
- [ ] Tag `v0.2-diffusion`

### Phase 3 — Experiment matrix (May 20–26)

- [ ] Run cells: `aug_type {none, classical, gen, classical+gen} × ratio {0,1,5,10}× × test {clean, noisy, cross-source}`, 3 seeds each
- [ ] Paired bootstrap CIs on per-class F1
- [ ] Fairness audit: per-class FNR (esp. `belly_pain`), cross-source generalization gap
- [ ] Fill `ethics/MODEL_CARD.md`, `ethics/DATASHEET.md`, `ethics/bias_audit.md` with real numbers
- [ ] Tag `v0.3-results`

### Phase 4 — Report + final polish (May 26–29)

- [ ] Populate `report/Report.tex` (NIPS 2017 template) with all sections including a substantive Ethics & Limitations section
- [ ] Final figures: training curves, confusion matrices, FAD-vs-F1 scatter, FNR-by-class bars
- [ ] Tag `v1.0-submission`; archive Zenodo DOI

## Experiment design

| Axis | Values |
|---|---|
| Augmentation type | none / classical / generative / classical+generative |
| Synthetic-to-real ratio (rare classes only) | 0× / 1× / 5× / 10× |
| Test condition | clean / additive babble noise (MUSAN, SNR ∈ {0, 5, 10} dB) / cross-source held-out |
| Seeds | {0, 1, 2} |

**Primary metric:** macro-F1.
**Headline secondary:** per-class recall on `belly_pain` and `burping` (rarest two).
**Reporting:** all numbers as mean ± paired-bootstrap 95% CI across seeds.
**Leakage protection:** splits fixed by source/file *before* synthesis; synthesis only uses train; pytest assertion enforces no test-file appears in any synthesis manifest.

## Risks & mitigations

1. **Solo vs. 2-per-team rule.** Email instructor Day 1. If denied, recruit partner before May 6 or fall back to PriorMDM.
2. **Donateacry size (~457 clips) is small for diffusion.** Pretrain unconditionally on a public Kaggle/Zenodo cry compilation, then condition-fine-tune on donateacry.
3. **Synthetic→test leakage.** Splits fixed pre-synthesis; pytest assertion in `tests/test_no_leakage.py`.
4. **AWS budget.** Diffusion ≤30M params, ≤6h on `ml.g5.xlarge`. Classifier on local MPS. SageMaker only for the matrix sweep.
5. **Reading HW collision.** HW front-loaded; submitted by May 4 to leave buffer.
6. **Null result.** A clean null result with a strong fairness audit is still publishable — the paper framing accommodates it.

## Commit & release workflow

- `main` is published; work on short-lived feature branches (`feat/*`, `exp/*`).
- Commit at end of every productive session; tag at each milestone.
- Commit messages: `area: imperative summary` (e.g., `eval: per-class F1 + ECE`).
- Never commit raw audio, AWS creds, or large checkpoints (push to S3, link from README).
