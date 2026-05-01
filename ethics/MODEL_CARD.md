# Model Card — Infant Cry Classifier (with optional generative augmentation)

Following *Mitchell et al., "Model Cards for Model Reporting" (2019)*.

> **Status:** scaffolded. Metric fields will be populated at `v0.1-baseline` and refined through `v0.3-results`.

## Model details

- **Person/org developing the model:** Syed Omer Shah, UB CSE 4/555 final project, Spring 2026.
- **Model date:** _(to be set at v1.0)_
- **Model version:** _(milestone tag)_
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

### Generative arm + multi-seed matrix (Phase 3)

_(populated after `python -m experiments.run_matrix`)_

- Per-class F1 / recall under generative and classical+generative aug, at
  synth-to-real ratios {0,1,5,10}, three seeds each.
- ECE per condition.
- Robustness gap (clean vs. additive babble noise; cross-source if a second
  corpus is integrated).
- FAD proxy and synthetic-class consistency for the diffusion model.

## Ethical considerations

- **Sensitive population:** infants. The model output should never be the sole basis for any clinical action.
- **Bias risks:** demographic skew of training data toward English-speaking, smartphone-equipped, Western caregivers — the model's accuracy on under-represented populations is unknown and likely worse.
- **Privacy:** no user audio is required for inference at evaluation time; if deployed, on-device inference is strongly recommended.
- **Mitigations:** generative augmentation is framed as a *fairness intervention* for rare classes — its effect on rare-class recall is the headline metric. Per-class metrics are always reported alongside aggregate metrics. Cross-source held-out evaluation is run as a proxy for demographic-shift robustness.

## Caveats and recommendations

- Treat as a **research artifact**, not a product.
- Do not deploy without (a) clinical validation, (b) on-device privacy review, (c) a UI framing that prevents the user from interpreting the output as a medical diagnosis, and (d) a per-deployment fairness audit on the actual user population.
