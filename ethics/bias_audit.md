# Bias Audit

> **Status:** scaffolded. Will be filled in after EDA (Phase 0) and updated at each milestone.

## Second-corpus integration and a data-quality finding

We integrated the public *Donate-A-Cry-Augmented* corpus on Kaggle to address the burping rare-class shortage (only 6 clips in donateacry train). On md5-content comparison we found that:

- The Kaggle `hungry` class is byte-identical to donateacry `hungry` (382 / 382).
- For each Kaggle rare class, only a fraction of clips are truly new content; the rest are duplicates of donateacry clips, **including duplicates labeled with a different class**. Specifically: 73 of 118 Kaggle "burping" clips are byte-identical to donateacry clips of other classes (mostly `hungry`); 66 / 127 belly_pain, 103 / 136 tired, 103 / 138 discomfort.

Mislabeled duplicates are filtered out by the manifest builder at build time. Only clips whose md5 does not appear anywhere in donateacry (any class) are added to the training pool. Final unique additions: 41 belly_pain, 37 burping, 3 discomfort, 4 tired, 0 hungry (85 total). After integration, the training pool grows from 320 to 405 clips and rare-class train counts grow from {belly_pain: 11, burping: 6} to {belly_pain: 52, burping: 43}.

Validation and test splits remain donateacry-real-only, so all reported metrics are still measured on the original held-out distribution.

This data-quality issue with the Kaggle redistribution is itself a fairness-relevant finding: a deployed model trained naively on the Kaggle CSV labels would learn a correlation between the burping label and what is acoustically a hungry cry. Safety-critical class labels in public corpora warrant content-level deduplication before use.

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

## Numeric findings — v1.0 final matrix (36 cells: 4 aug × 3 recipes × 3 seeds)

The fairness intervention works for one rare class but not the other. The best
cell (`generative + weighted_ce`) catches the test burping clip on 2 of 3
seeds (mean recall 0.667), and macro-F1 is 33% above baseline. No cell across
the matrix recovers belly_pain at the argmax level, despite the diffusion
probe identifying 93% of synthetic belly_pain as belly_pain.

| Arm | Recipe | macro-F1 | belly_pain rec. | burping rec. | ECE |
|---|---|---:|---:|---:|---:|
| none | weighted_ce | 0.194 | 0.000 | 0.000 | 0.285 |
| none | balanced | 0.191 | 0.111 | 0.000 | 0.201 |
| none | focal | 0.213 | 0.000 | 0.000 | 0.354 |
| classical | weighted_ce | 0.180 | 0.000 | 0.000 | 0.418 |
| classical | balanced | 0.205 | 0.000 | 0.000 | 0.285 |
| classical | focal | 0.187 | 0.000 | 0.000 | 0.345 |
| **generative** | **weighted_ce** | **0.258** | 0.000 | **0.667** | 0.352 |
| generative | balanced | 0.195 | 0.000 | 0.333 | 0.226 |
| generative | focal | 0.238 | 0.000 | 0.333 | 0.333 |
| classical+generative | weighted_ce | 0.208 | 0.000 | 0.333 | 0.286 |
| classical+generative | balanced | 0.171 | 0.000 | 0.000 | 0.207 |
| classical+generative | focal | 0.182 | 0.000 | 0.000 | 0.371 |

**Fairness reading.** Generative augmentation succeeds where classical
augmentation fails for the burping class (3/4 generative cells produce
non-zero burping recall; 0/18 non-generative cells do). For belly_pain,
no augmentation type or recipe combination recovers argmax-level recall:
even with 52 real+synthetic training clips after the kaggle integration,
the downstream classifier still defaults away from belly_pain on every
test instance. The 11 real belly_pain clips are a hard floor on what
augmentation alone can fix at this data scale, and the project's deployment
recommendation is correspondingly to disable a "pain" alert until a third
corpus is integrated.

## Numeric findings — v0.3 baseline matrix (donateacry-only, 4 arms × 3 seeds)

The full augmentation matrix at synthetic-to-real ratio 10× shows that none
of the four arms flips the classifier's argmax on the safety-critical rare
classes. This is the headline fairness finding of the project.

| Arm | macro-F1 | belly_pain rec. | burping rec. | ECE (mean) |
|---|---:|---:|---:|---:|
| none                 | 0.183 | 0.00 | 0.00 | 0.48 |
| classical            | 0.183 | 0.00 | 0.00 | 0.48 |
| generative           | 0.182 | 0.00 | 0.00 | 0.37 |
| classical+generative | 0.183 | 0.00 | 0.00 | 0.39 |

The fairness intervention (generative augmentation targeted at rare classes)
delivers a measurable but partial benefit: it reduces calibration error by
about 10 points without changing the safety-critical FNR. In the user-facing
sense, this means a deployed system would be less overconfident on its
mispredictions, but would still miss every pain cry. The intended
intervention does not cross the threshold needed to actually improve safety
behavior at this data scale.

## Numeric findings — Phase 1 baselines (seed=0, 30 epochs)

The two naive baselines below are the empirical motivation for the project.
Both achieve high overall accuracy by predicting the majority class
(`hungry`) for every input, and zero recall on the safety-critical rare
classes. This is exactly the failure mode the per-class FNR audit and the
generative augmentation extension are designed to address.

| Augmentation | macro-F1 | accuracy | belly_pain recall | burping recall | ECE |
|---|---:|---:|---:|---:|---:|
| none | 0.183 | 0.841 | **0.00** | **0.00** | 0.48 |
| classical | 0.183 | 0.841 | **0.00** | **0.00** | 0.51 |

Aggregate accuracy here is *worse* than per-class recall in the ethics-relevant
sense: a parent-facing app reporting 84% accuracy with zero ability to detect
pain would be actively dangerous. The fairness intervention reported in
Phase 3 (generative augmentation targeted at rare classes) is evaluated
against this baseline.

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
