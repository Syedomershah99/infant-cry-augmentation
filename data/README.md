# Data

This folder holds **manifests only**. Raw clips are gitignored and downloaded via scripts.

## Sources

| Source | Clips | Classes | License | Use |
|---|---|---|---|---|
| [donateacry-corpus](https://github.com/gveres/donateacry-corpus) | ~457 | belly_pain, burping, discomfort, hungry, tired | MIT (corpus) | Train / val / test (primary) |
| Public Kaggle / Zenodo infant-cry compilations (TBD per-source CC license) | varies | varies | CC-* | Cross-source held-out test + unconditional diffusion pretraining |
| [MUSAN](https://www.openslr.org/17/) babble subset | ~ | n/a | Apache 2.0 | Additive-noise robustness eval |
| [ESC-50](https://github.com/karolpiczak/ESC-50) ambient subset | varies | n/a | CC BY-NC 3.0 | (Optional) ambient-noise eval — note non-commercial |

See [`../ethics/consent_provenance.md`](../ethics/consent_provenance.md) for per-source consent basis.

## Splits

- Splits are **source-stratified** and fixed by file before any model is trained or any synthetic clip is generated.
- A pytest assertion (`tests/test_no_leakage.py`) verifies no test file ever appears in a synthesis manifest or training augmentation pool.

## Manifest schema (`data/manifests/*.csv`)

| column | description |
|---|---|
| `filepath` | relative path under `data/raw/` |
| `label` | one of {belly_pain, burping, discomfort, hungry, tired} (primary) |
| `source` | corpus name (e.g., `donateacry`, `kaggle_X`) |
| `split` | one of {train, val, test, cross_test} |
| `duration_s` | clip duration (seconds) |
| `sample_rate` | original sample rate |
| `seed_for_split` | RNG seed used to assign this row to its split |
| `consent_basis` | short tag pointing into `ethics/consent_provenance.md` |
