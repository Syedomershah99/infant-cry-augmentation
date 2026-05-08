# Consent & Provenance

For each upstream data source, this file records the provenance and the consent basis under which clips are used in this project.

## Per-source records

### donateacry-corpus
- **Origin:** community-uploaded corpus aggregated at <https://github.com/gveres/donateacry-corpus>.
- **Upstream license:** MIT (corpus repository).
- **Consent basis:** parents/caregivers voluntarily uploaded clips through the original Donate-A-Cry application; the corpus is published as a public research dataset.
- **Use in this project:** training (267 hungry + 53 rare-class clips), validation (68 clips), test (69 clips), and class-conditional fine-tuning of the diffusion model.
- **Redistribution in this repo:** **none.** Manifests only; clips fetched from upstream.

### Donate-A-Cry-Augmented (Kaggle)
- **Origin:** Kaggle redistribution of donateacry-corpus with programmatic augmentations of rare-class clips. Downloaded by the user as `archive.zip` and unzipped under `data/raw/kaggle_archive/`.
- **Upstream consent basis:** the Kaggle redistribution inherits the donateacry consent basis (parents/caregivers who voluntarily uploaded to the original Donate-A-Cry application).
- **Use in this project:** only clips whose md5 content hash is **not** present anywhere in the donateacry corpus are retained, and they are added exclusively to the train split. The 0 hungry / 41 belly_pain / 37 burping / 3 discomfort / 4 tired = 85 unique clips supplement the train pool from 320 to 405. Validation and test remain donateacry-real-only.
- **Data-quality flag:** before content deduplication, 73 / 118 burping clips, 66 / 127 belly_pain clips, 103 / 136 tired clips, and 103 / 138 discomfort clips in this redistribution were byte-identical to donateacry clips of *other* (mostly hungry) classes. We treat this as label noise rather than as new content, and document it as a fairness-relevant finding in `bias_audit.md`.
- **Redistribution in this repo:** **none.** Manifests only; clips fetched from the upstream Kaggle source.

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
