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
