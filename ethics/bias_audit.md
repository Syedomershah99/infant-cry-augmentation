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

## Numeric findings (to populate)

_(Per-class F1 / recall / FNR will be inserted here at v0.3-results, per condition.)_
