# Bias Audit

> **Status:** scaffolded. Will be filled in after EDA (Phase 0) and updated at each milestone.

## Audited slices

| Slice | Why it matters | Where reported |
|---|---|---|
| Class | Rare classes (`belly_pain`, `burping`) are the safety-critical / fairness-critical cells | per-class F1 + recall in all results tables |
| Source | Cross-source generalization is a proxy for demographic robustness | dedicated cross-source held-out test split |
| Noise condition | Real-world deployments are noisy; underrepresented populations may face worse SNR | noise-overlay eval at SNR ∈ {0, 5, 10} dB |
| Clip duration | Short clips may be harder to classify; duration distribution may correlate with class | duration-binned F1 (sanity slice) |

## Known and suspected biases (pre-results)

- **Class imbalance.** donateacry is dominated by `hungry`. Models without explicit intervention will exhibit weak `belly_pain` recall — exactly the failure mode that matters most clinically.
- **Demographic skew.** Parent-uploaded smartphone corpora skew toward English-speaking, Western, connected caregivers. We do not have demographic metadata to audit this directly; we use **cross-source generalization gap** as a proxy.
- **Label noise.** Labels are parent-reported. `belly_pain` vs. `discomfort` is plausibly noisy — adjacent classes share acoustic features.
- **Recording-channel bias.** Smartphone-microphone bias may dominate the spectral statistics; performance on a clinical-microphone corpus is unknown.

## Mitigations applied in this project

- **Per-class reporting** is mandatory in every results table — aggregate metrics never appear without per-class breakdowns.
- **Source-stratified splits.** No source crosses train→test boundaries when the cross-source held-out is in use.
- **Generative augmentation as a fairness intervention.** Synthetic samples are added *only to rare classes in train* and the effect is measured on rare-class recall.
- **No re-identification claims.** The model produces a class label, never a speaker identity.
- **Pytest assertion** that no test-split file ever appears in any synthesis or augmentation manifest.

## Limitations of this audit

- Demographic features of the upstream corpora are not annotated, so direct demographic-fairness slicing is not possible.
- Cross-source generalization is a *proxy* for demographic robustness — gaps may be due to source-specific recording conditions rather than population differences. The audit should be read as a lower bound on population-level concerns.

## Numeric findings — donateacry-corpus (Phase 0 EDA)

Total: 457 clips, seed=42 stratified split.

| Class | All | Train | Val | Test | % of corpus |
|---|---:|---:|---:|---:|---:|
| hungry | 382 | 267 | 57 | 58 | 83.6% |
| discomfort | 27 | 19 | 4 | 4 | 5.9% |
| tired | 24 | 17 | 4 | 3 | 5.3% |
| belly_pain | 16 | 11 | 2 | 3 | 3.5% |
| burping | 8 | 6 | 1 | 1 | 1.8% |

**Imbalance ratio** (largest:smallest) = 47.75. The two rarest classes (`burping`, `belly_pain`) together account for 5.3% of the corpus. This is exactly the regime the project's research question targets.

### Methodological flags raised by the split

- **Burping has 1 clip in val and 1 in test.** Single-clip evaluation is statistically unreliable (recall is binary 0 or 1). For the burping class specifically, results will be reported with a clearly labeled note; if time permits, a 5-fold leave-one-out evaluation on the rare classes (combining val+test) will be added as a robustness check.
- **Belly_pain has 2 in val, 3 in test.** Slightly better but still high-variance; same robustness note applies.

### Demographic metadata in the corpus

The donateacry filename convention encodes `<gender>` (m/f) and an `<age>` field. We parse these into `gender` and `weeks` columns in the manifest to enable demographic slicing. Two caveats:

1. These fields are caregiver-self-reported and may be missing/incorrect.
2. Gender of an infant is not a population-fairness axis in the same sense that race or socioeconomic status would be — the parameter we *cannot* audit (because it's not annotated upstream) is parental population, recording region, and recording channel. The cross-source held-out split is the only proxy we have for those.

### Filename↔directory label disagreements

The manifest also carries a `filename_label_mismatch` flag — set when the trailing class code in the filename disagrees with the directory the file lives in. This is a sanity check against upstream label noise; counts will be reported in the EDA notebook.
