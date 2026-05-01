# Datasheet — Infant Cry Augmentation Project

Following the framework of *Gebru et al., "Datasheets for Datasets" (2018)*. This datasheet covers the **derived** dataset used in this project (manifests + splits over public sources), not the original corpora — see each source's own datasheet/README for upstream details.

> **Status:** scaffolded. Numeric fields will be populated after `data/manifests/` is built and before `v0.1-baseline`.

## Motivation

- **For what purpose was the dataset created?** To enable a controlled study of class-conditional generative augmentation for severely class-imbalanced infant cry classification, with explicit fairness/robustness reporting on rare classes.
- **Who created the dataset?** Syed Omer Shah, as the final project for CSE 4/555 (UB, Spring 2026). The underlying recordings are from third-party public sources (see `consent_provenance.md`).
- **Funding?** Coursework; AWS credits used for compute.

## Composition

- **What do the instances represent?** Short audio clips of infant cries, with one of five class labels (belly_pain, burping, discomfust, hungry, tired) plus source and split metadata.
- **How many instances?** _(TBD after manifest build)_
- **What data does each instance consist of?** Raw audio file (downloaded separately, not redistributed in this repo) + a row in a manifest CSV.
- **Is there a label or target?** Yes — categorical class label inherited from the upstream source.
- **Are relationships between instances made explicit?** Yes — `source` and `split` columns; cross-source held-out split is preserved.
- **Are recommended splits provided?** Yes — see `data/manifests/`.
- **Are there errors, sources of noise, or redundancies?** Upstream labels are parent-reported and unverified; sample rates and recording conditions vary. The cross-source split exists specifically to surface this.
- **Does the dataset contain confidential / sensitive data?** Yes — infant audio is potentially sensitive PII (voice biometric). Project does not attempt re-identification and treats all clips as already-public-with-consent per the upstream source's terms.

## Collection process

- **How was the data acquired?** Downloaded via scripts from upstream public repositories (donateacry-corpus on GitHub, Kaggle / Zenodo as applicable). No new recordings were collected.
- **What mechanisms / sampling strategy?** All available clips from each source are pulled; no subsampling beyond split assignment.
- **Who was involved?** Original parents/caregivers uploaded clips to upstream sources under their respective consent regimes.
- **Time frame?** Spring 2026.
- **Ethical review?** No new IRB; project relies on upstream consent. See `consent_provenance.md`.

## Preprocessing / cleaning / labeling

- **Was any preprocessing done?** Resampling to 16 kHz, fixed-duration crops, log-mel spectrogram extraction. Performed at training time, not committed.
- **Was raw data saved?** Raw clips remain at their upstream source; only manifests live in this repo.

## Uses

- **Has the dataset been used for any tasks already?** This is its first integrated use.
- **What other tasks could it be used for?** Audio classification benchmarking, class-imbalance research, fairness audits.
- **Are there tasks for which the dataset should not be used?** **Clinical diagnosis.** This dataset is **not** a clinical-grade resource. Do not use it to triage or diagnose actual infants without independent clinical validation.

## Distribution

- **Will the dataset be distributed?** No — only manifests are distributed (in this repo). Raw clips must be obtained from the upstream sources directly under their respective licenses.

## Maintenance

- **Who is supporting / maintaining?** Syed Omer Shah (during the course; archived after submission).
- **Will the dataset be updated?** Manifests may be updated to fix split errors discovered during the project; the cross-source held-out split is frozen at `v0.1-baseline`.

## Known limitations & biases

_(Will be expanded in `bias_audit.md` after EDA.)_

- **Class imbalance:** donateacry is dominated by `hungry`; rare classes (`belly_pain`, `burping`) have on the order of dozens of clips each.
- **Demographic skew:** parent-uploaded corpora skew toward connected, English-speaking, Western caregivers.
- **Recording-condition skew:** smartphone microphones, varied background noise, varied SNR.
- **Label noise:** labels are parent-reported, not pediatrician-verified.
