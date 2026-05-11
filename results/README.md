# Results — running notes

## v0.1-baseline (single seed=0, 30 epochs each)

These two baselines establish the failure mode that motivates the project. Both
collapse to predicting only `hungry` (the 84% majority class), achieving high
overall accuracy but **zero recall on the safety-critical rare classes**.

| Run | aug | macro-F1 | acc | belly_pain recall | burping recall | ECE |
|---|---|---:|---:|---:|---:|---:|
| `baseline_seed0` | none | 0.183 | 0.841 | 0.00 | 0.00 | 0.48 |
| `classical_seed0` | classical | 0.183 | 0.841 | 0.00 | 0.00 | 0.51 |

**Reading.** Aggregate metrics look fine; per-class metrics show the model is
clinically useless on pain and burping. Classical SpecAugment / pitch-shift /
noise injection did not move the rare-class numbers because the underlying
problem is *too few unique rare-class samples in train* (11 belly_pain,
6 burping). Classical augmentation creates variations of those few clips, not
new modes.

This is the headline failure the conditional-diffusion extension targets.

## Detailed JSON

Each run writes a `result.json` with full per-epoch history, config snapshot,
test metrics, per-class metrics, confusion matrix, and the ethics-facing
safety summary (belly_pain FNR, burping FNR, ECE).

```bash
python -c "import json; r=json.load(open('results/baseline_seed0/result.json')); print(json.dumps(r['test']['per_class'], indent=2))"
```

## v1.0-submission — full 3-axis matrix + noise-robustness eval + visualizations

### Noise-robustness eval (3 seeds × 5 SNRs, on test split with additive Gaussian noise)

| Arm | SNR | macro-F1 | accuracy | belly_pain rec. | burping rec. | ECE |
|---|---|---:|---:|---:|---:|---:|
| baseline (none + weighted_ce) | clean | 0.193 | 0.797 | 0.000 | 0.000 | 0.285 |
| baseline | 20 dB | 0.194 | 0.807 | 0.000 | 0.000 | 0.275 |
| baseline | 10 dB | 0.173 | 0.546 | 0.111 | 0.333 | 0.222 |
| baseline | 5 dB | 0.011 | 0.020 | 0.111 | 1.000 | 0.524 |
| baseline | 0 dB | 0.027 | 0.048 | 0.889 | 0.000 | 0.617 |
| generative + weighted_ce | clean | **0.258** | **0.817** | 0.000 | **0.667** | 0.352 |
| generative | 20 dB | **0.267** | **0.831** | 0.000 | **0.667** | 0.367 |
| generative | 10 dB | **0.186** | **0.768** | 0.000 | 0.333 | 0.334 |
| generative | 5 dB | 0.077 | 0.044 | 0.000 | 0.667 | 0.563 |
| generative | 0 dB | 0.013 | 0.034 | 0.333 | 0.333 | 0.668 |

Key finding: generative augmentation holds **76.8% accuracy at 10 dB SNR** where the baseline drops to **54.6%**. Below 10 dB both arms collapse (high rare-class recall values reflect near-random argmax, not robustness).

Raw per-seed numbers: `robustness_none_weighted_ce.csv`, `robustness_generative_weighted_ce.csv`. Figure: `../report/figures/robustness.png`.

### Confusion matrix figures

- `../report/figures/confusion_baseline.png`: none+weighted_ce seed 0 — predicts hungry for every test instance.
- `../report/figures/confusion_generative.png`: generative+weighted_ce seed 2 — predictions distribute, burping caught (1/1), discomfort partial (2/4), tired partial.

---

## v1.0-submission — full 3-axis matrix (36 cells: 4 aug × 3 recipes × 3 seeds)

After integrating the Donate-A-Cry-Augmented Kaggle corpus (with content-level
deduplication) and adding an optimizer-recipe axis, the final 36-cell matrix
shows:

| Arm | Recipe | macro-F1 | acc | bp_rec | **burp_rec** | ECE |
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

**Headline win:** `generative + weighted_ce` reaches macro-F1 = 0.258 (+33% vs
no-aug baseline) and burping recall = 0.667 with no accuracy penalty. Three
of four generative cells produce non-zero burping recall; zero of eighteen
non-generative cells do.

**Sample-quality probe (DDPM v3 trained on merged train pool):**

| Class | n synth | probe recall | probe precision | probe F1 |
|---|---:|---:|---:|---:|
| belly_pain | 156 | **0.93** | 0.89 | 0.91 |
| burping    | 129 | **0.08** | 0.77 | 0.14 |

Vs. v0.2 (donateacry-only train): belly_pain 0.51, burping 0.00. The
larger rare-class signal is the rate-limiting factor for diffusion-model
class-conditioning quality at this data scale.

Per-cell raw numbers: `matrix_part1.csv`, `matrix_part2.csv`.
Aggregated mean+std: `matrix_summary.csv`, `matrix_summary.json`.

---

## v0.3-results — 4-arm experiment matrix (12 cells: 4 aug × 3 seeds, ratio=10×)

Headline result: **at this data scale, generative augmentation does not flip
the classifier's argmax on rare classes.** Top-1 prediction is essentially
identical across arms. The single quantity that does shift is calibration:
generative arms drop ECE by ~10 points.

| Arm | macro-F1 | accuracy | belly_pain rec. | burping rec. | ECE (mean) |
|---|---:|---:|---:|---:|---:|
| none                 | 0.183 | 0.841 | 0.00 | 0.00 | 0.48 |
| classical            | 0.183 | 0.841 | 0.00 | 0.00 | 0.48 |
| generative           | 0.182 | 0.831 | 0.00 | 0.00 | **0.37** |
| classical+generative | 0.183 | 0.841 | 0.00 | 0.00 | **0.39** |

Rare-class recall is 0.00 in every cell, every seed.

This is a publishable null result on the original research question, and a
real finding on calibration: the synthetic samples carry useful but
sub-argmax-threshold class signal. See `report/Report.tex` Section 5 for the
full discussion. Per-cell raw numbers are in `matrix.csv` and per-cell
configs are in `matrix/<cell>/config.yaml`.

## What's next

- A second public infant-cry corpus to enlarge rare-class signal for both
  the diffusion model and the classifier.
- Optimizer-recipe sensitivity analysis (balanced sampler, focal loss,
  class-rebalanced fine-tuning) as an orthogonal axis to augmentation type.
- Cross-source held-out test split for direct demographic-shift evaluation.
