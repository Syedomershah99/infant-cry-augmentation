# Consent & Provenance

For each upstream data source, this file records the provenance and the consent basis under which clips are used in this project.

> **Status:** scaffolded. Will be populated as sources are pulled in Phase 0.

## Per-source records

### donateacry-corpus
- **Origin:** community-uploaded corpus aggregated at <https://github.com/gveres/donateacry-corpus>.
- **Upstream license:** MIT (corpus repository).
- **Consent basis:** parents/caregivers voluntarily uploaded clips through the original Donate-A-Cry application; the corpus is published as a public research dataset.
- **Use in this project:** training, validation, test, and conditional fine-tuning of the diffusion model.
- **Redistribution in this repo:** **none.** Manifests only; clips fetched from upstream.

### Cross-source corpus(es) — TBD
- **Origin:** _(filled per source as added — Kaggle / Zenodo)_
- **Upstream license:** _(must be CC-BY or compatible; non-commercial sources flagged below)_
- **Consent basis:** _(quoted from upstream source's documentation)_
- **Use:** cross-source held-out test only, and unconditional diffusion pretraining.
- **Redistribution in this repo:** **none.**

### MUSAN babble subset (noise robustness)
- **Origin:** <https://www.openslr.org/17/>
- **Upstream license:** Apache 2.0.
- **Consent basis:** publicly released speech/noise corpus.
- **Use:** additive-noise overlay for robustness eval only. No infant clips are derived from MUSAN.

### ESC-50 ambient subset (optional)
- **Origin:** <https://github.com/karolpiczak/ESC-50>
- **Upstream license:** **CC BY-NC 3.0** — non-commercial.
- **Consent basis:** Freesound contributor licensing.
- **Use:** non-commercial research use only. Flagged here so any follow-up commercial deployment must drop ESC-50.

## Practices

- **No re-identification.** This project produces class labels, not speaker identities. We do not attempt voiceprint extraction.
- **No raw clips in this repo.** Only manifests are committed; clips remain at their upstream source.
- **Removal requests.** If a contributor to any upstream source requests removal of their clip, the corresponding row is dropped from our manifests and any cached spectrogram is regenerated. (For this course project, follow upstream removal channels first; we will mirror.)
- **No PII fields.** Manifests carry only `filepath`, `label`, `source`, `split`, `duration_s`, `sample_rate`, `seed_for_split`, `consent_basis`. No names, no caregiver IDs, no geolocation.
