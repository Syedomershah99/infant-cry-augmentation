# Model Card — Infant Cry Classifier (with optional generative augmentation)

Following *Mitchell et al., "Model Cards for Model Reporting" (2019)*.

## Model details

- **Person/org developing the model:** Syed Omer Shah and Collin Murphy, Group 14, UB CSE 4/555 final project, Spring 2026.
- **Model date:** May 2026.
- **Model version:** v1.0-submission (best cell: `generative + weighted_ce`, seed=0).
- **Model type:** Audio classifier — log-mel CNN baseline, optionally a fine-tuned Audio Spectrogram Transformer (AST) or PANNs CNN14. Augmented variants additionally use a class-conditional DDPM for synthetic spectrogram generation.
- **License:** MIT (code).
- **Citation:** see repo `README.md`.

## Intended use

- **Primary intended uses:** **Research artifact** for studying class-imbalance interventions in audio classification, and an educational baseline for pattern-recognition coursework.
- **Primary intended users:** ML researchers, students, and developers prototyping infant-audio interfaces.
- **Out-of-scope uses:**
  - **Clinical triage or diagnosis** of any infant.
  - Surveillance / scoring of caregivers.
  - Any deployment without independent clinical validation, on-device privacy review, and a clear "decision support, not diagnostic" UI framing.

## Factors

- Recording channel (microphone type, sample rate)
- Background-noise environment
- Cry class (especially the rare classes `belly_pain` and `burping`)
- Source corpus (donateacry vs. cross-source)

These are the slices on which evaluation is reported.

## Metrics

- **Primary:** macro-F1, per-class recall (especially `belly_pain` — the safety-critical class).
- **Calibration:** Expected Calibration Error (15-bin).
- **Robustness gap:** F1(clean) − F1(noisy / cross-source).
- **Reporting:** mean ± paired-bootstrap 95% CI over 3 seeds.

## Evaluation data

donateacry-corpus held-out split + cross-source held-out split + noisy-overlay variants. See `data/README.md`.

## Training data

donateacry-corpus train split. Optional pretraining for the diffusion model on a public Kaggle/Zenodo cry compilation (sources listed in `consent_provenance.md`). Synthetic samples are added only to the training pool; never to val/test.

## Quantitative analyses

### Baselines (Phase 1, single seed=0, donateacry test split, n=69)

| Augmentation | macro-F1 | accuracy | belly_pain recall | burping recall | ECE |
|---|---:|---:|---:|---:|---:|
| none | 0.183 | 0.841 | **0.00** | **0.00** | 0.48 |
| classical (SpecAug + noise + time-shift) | 0.183 | 0.841 | **0.00** | **0.00** | 0.51 |

Both naive baselines collapse to predicting the majority class (`hungry`,
84% of test). Aggregate accuracy is misleading; per-class recall on the
safety-critical `belly_pain` class is zero.

### Final augmentation × optimizer-recipe matrix (v1.0: 4 arms × 3 recipes × 3 seeds = 36 cells)

| Arm | Recipe | macro-F1 | accuracy | belly_pain rec. | burping rec. | ECE |
|---|---|---:|---:|---:|---:|---:|
| none | weighted_ce | 0.194 | 0.797 | 0.000 | 0.000 | 0.285 |
| none | balanced | 0.191 | 0.507 | 0.111 | 0.000 | 0.201 |
| none | focal | 0.213 | 0.787 | 0.000 | 0.000 | 0.354 |
| classical | weighted_ce | 0.180 | 0.807 | 0.000 | 0.000 | 0.418 |
| classical | balanced | 0.205 | 0.638 | 0.000 | 0.000 | 0.285 |
| classical | focal | 0.187 | 0.778 | 0.000 | 0.000 | 0.345 |
| **generative** | **weighted_ce** | **0.258** | **0.816** | 0.000 | **0.667** | 0.352 |
| generative | balanced | 0.195 | 0.585 | 0.000 | 0.333 | 0.226 |
| generative | focal | 0.238 | 0.763 | 0.000 | 0.333 | 0.333 |
| classical+generative | weighted_ce | 0.208 | 0.792 | 0.000 | 0.333 | 0.286 |
| classical+generative | balanced | 0.171 | 0.507 | 0.000 | 0.000 | 0.207 |
| classical+generative | focal | 0.182 | 0.836 | 0.000 | 0.000 | 0.371 |

**Recommended deployment configuration:** `generative + weighted_ce` is the
best cell. It achieves the highest macro-F1 (0.258), the highest non-zero
burping recall (0.667 across seeds), and preserves majority-class accuracy
(0.816). It does NOT recover belly_pain (still 0.000), so any deployment
should explicitly disable a "pain" alert until rare-class recall improves
on a larger or genuinely cross-source corpus.

### Sample-quality probe (DDPM v3 trained on merged train pool)

| Class | n synth | probe recall | probe precision | probe F1 |
|---|---:|---:|---:|---:|
| belly_pain | 156 | 0.93 | 0.89 | 0.91 |
| burping    | 129 | 0.08 | 0.77 | 0.14 |

The probe's belly_pain recognition (0.93) is far above the downstream
classifier's belly_pain recall (0.000 in all cells). This gap suggests an
iterative refinement loop (classifier-scored sample re-training) is the most
promising follow-up.

### Earlier baselines (v0.3, donateacry-only, 4 arms × 3 seeds, 12 cells)

| Arm | macro-F1 | accuracy | belly_pain rec. | burping rec. | ECE (mean) |
|---|---:|---:|---:|---:|---:|
| none                 | 0.183 | 0.841 | 0.00 | 0.00 | 0.48 |
| classical            | 0.183 | 0.841 | 0.00 | 0.00 | 0.48 |
| generative           | 0.182 | 0.831 | 0.00 | 0.00 | **0.37** |
| classical+generative | 0.183 | 0.841 | 0.00 | 0.00 | **0.39** |

Rare-class recall is zero in every cell of the matrix. Generative
augmentation reduces ECE by approximately 10 points without changing the
classifier's argmax. The synthetic samples contain useful class signal (a
class-consistency probe identifies 51% of synthetic belly_pain as belly_pain)
but the signal is below the classifier's decision threshold under
class-weighted CE alone.

### Sample-quality probe (30-epoch DDPM, CFG=2.0)

| Class | n synth | probe recall | probe precision | probe F1 |
|---|---:|---:|---:|---:|
| belly_pain | 110 | 0.51 | 0.88 | 0.64 |
| burping    | 60  | 0.00 | 0.00 | 0.00 |

The asymmetry is consistent with diffusion training data: belly_pain has 11
real training clips, burping has 6, and below this threshold the DDPM did
not learn a class-discriminative manifold for burping.

## Ethical considerations

- **Sensitive population:** infants. The model output should never be the sole basis for any clinical action.
- **Bias risks:** demographic skew of training data toward English-speaking, smartphone-equipped, Western caregivers — the model's accuracy on under-represented populations is unknown and likely worse.
- **Privacy:** no user audio is required for inference at evaluation time; if deployed, on-device inference is strongly recommended.
- **Mitigations:** generative augmentation is framed as a *fairness intervention* for rare classes — its effect on rare-class recall is the headline metric. Per-class metrics are always reported alongside aggregate metrics. Cross-source held-out evaluation is run as a proxy for demographic-shift robustness.

## Caveats and recommendations

- Treat as a **research artifact**, not a product.
- Do not deploy without (a) clinical validation, (b) on-device privacy review, (c) a UI framing that prevents the user from interpreting the output as a medical diagnosis, and (d) a per-deployment fairness audit on the actual user population.
