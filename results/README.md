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

## What's next (Phase 2)

- Train class-conditional DDPM on log-mel patches.
- Sample N synthetic spectrograms per rare class.
- Add to `synth_train.csv` and rerun with `aug_type: generative` and
  `aug_type: classical+generative`.
- Compare rare-class recall and macro-F1 against the two baselines above.
