# ML4H 2026 submission metadata

Submission portal: ML4H 2026 OpenReview venue (link goes live when the CFP opens — typically July/August). Track us via <https://ml4health.github.io>.

Source tarball to upload (compiled with the official PMLR/JMLR style):
`report/infant-cry-augmentation-ml4h.tar.gz` (1.7 MB, 18 files)

The tarball contains the anonymized LaTeX source (`main.tex`), pre-built bibliography (`main.bbl` + `main.bib`), the PMLR/JMLR style files (`jmlr.cls`, `jmlrutils.sty`), and 12 figure PNGs in `figures/`. No PDF in the tarball — OpenReview compiles the source. Verified to compile end-to-end in an isolated directory with Tectonic.

---

## Track choice

**Proceedings track** (full paper, archival; published in PMLR after camera-ready).

- 8 pages of main content (refs/appendix do not count).
- Two-column 10pt PMLR/JMLR format (`\documentclass[pmlr,twocolumn,10pt]{jmlr}`).
- Our current rendering: 7 pages of main content + 1 page of references = 8 pages total. Inside limit.
- Findings track (4-page non-archival) is the fallback if reviewers ask for shorter; we do not start there because our content (36-cell matrix + noise sweep + ratio sweep + AST comparison) needs the 8-page room.

## Why ML4H is the right venue

- The paper is a clinical-adjacent ML paper: rare-class fairness intervention on infant cry classification, per-class FNR as the headline metric, datasheet + model card alongside accuracy. That is the audience ML4H is built for.
- The TMLR desk reject was on audience criterion — the work is applied/clinical-leaning rather than methods-novelty-leaning, which ML4H rewards.
- ML4H accepts incremental-but-rigorous applied work as long as the clinical framing and reporting hygiene are right. Both are central to this paper.

---

## Before you submit

- [ ] **OpenReview profiles** for both authors (real names + affiliation). Profiles are hidden from reviewers until decision but ML4H requires complete profiles for editorial routing.
- [ ] Both author profiles linked to the submission in OpenReview.
- [ ] **CRITICAL — public-repo hygiene:** the GitHub repo `Syedomershah99/infant-cry-augmentation` is currently public. ML4H is double-blind. Before submitting, either (a) make the repo **private** for the review window, or (b) host the supplementary at <https://anonymous.4open.science/> and reference that URL in the supplementary statement instead of the GitHub URL. Do **not** point reviewers at a public repo with author identities in commit history — this was a real risk factor at TMLR and we should not repeat it.
- [ ] Identity-leak scan run on `main.tex` and `main.bib` — clean (only the donateacry-corpus GitHub citation remains, which is the public dataset reference, not author identity).
- [ ] `main.bbl` is in the tarball (so OpenReview skips BibTeX).
- [ ] Author block reads `Anonymous Authors / Anonymous Institution / anonymous@anon` — auto by current source.
- [ ] Code/data paragraph in §1 Contributions says "repository URL withheld for double-blind review" — verified.
- [ ] No "Under review as submission to TMLR" footer (verified: jmlr.cls renders "Proceedings of Machine Learning Research TBD:1-9, 2026 / Machine Learning for Health (ML4H) Symposium 2026").

---

## OpenReview submission form (fields to fill at submission time)

### Title

```
Class-Conditional Generative Augmentation for Rare-Class Infant Cry Classification
```

### Abstract

```
Infant cry classification could inform neonatal triage and parent-support apps, but its safety profile is dominated by the rarest classes (such as pain), which are also the most under-represented in public corpora. In the widely-used donateacry-corpus, hungry accounts for 84% of clips and burping for 1.8%; a naive log-mel CNN reaches 84% aggregate accuracy on this corpus by collapsing onto the majority class, missing every safety-critical case. We treat class-conditional denoising diffusion (DDPM) as a targeted fairness intervention for the rare classes, paired with a 3-way optimizer-recipe ablation (class-weighted cross-entropy, balanced sampler, focal loss). Integrating a second widely-redistributed Kaggle corpus surfaces a fairness-relevant data-quality issue: many of the redistribution's labeled rare-class clips are byte-identical to clips of other classes; we deduplicate at content level before any model sees them. We report a controlled 4 x 3 x 3-seed augmentation matrix (36 cells) plus a noise-robustness sweep, a synth-to-real ratio sweep, and a backbone comparison against a frozen AudioSet-pretrained AST linear probe. The best cell, generative + weighted_ce, attains macro-F1 = 0.258 (vs. 0.194 baseline; +33%) and burping recall = 0.667 (vs. 0.000 across all 18 non-generative cells) while preserving 81.6% accuracy. Under additive noise the generative arm holds 76.8% accuracy at 10 dB SNR where the baseline drops to 54.6%. The ratio sweep exposes a non-monotonic threshold effect: 1x and 2x synthetic ratios behave like the baseline; 3x crosses an argmax threshold and burping recall jumps from 0 to 0.667. The AST comparison shows that pretraining and DDPM augmentation contribute orthogonally: pretraining recovers belly_pain that our small CNN never does, while the DDPM is the only intervention that recovers burping.
```

### TL;DR / Short summary

```
A class-conditional DDPM in mel space, used as a targeted rare-class fairness intervention on the donateacry-corpus, recovers burping (recall 0 -> 0.667) and gains +33% macro-F1 over the no-augmentation baseline with no accuracy penalty and a 22-point noise-robustness gain at 10 dB SNR. A pretrained AST backbone is complementary (it recovers belly_pain that the DDPM does not). We also surface a fairness-relevant cross-class label-noise issue in the most widely redistributed Kaggle infant-cry corpus.
```

### Keywords

```
class imbalance, fairness, data augmentation, denoising diffusion models, audio classification, infant cry, clinical machine learning, fairness audits, model cards, datasheets
```

### Primary area

Pick the closest ML4H subject area. Likely options on the form: *Health applications, audio/speech*, or *Fairness and accountability in clinical ML*. Either fits; choose the one with more "Applications" volume since this is an application paper with a fairness lens, not a fairness-theory paper.

### Conflicts

None known. Independent academic research; no industry funding; no proprietary data.

### Compute/data declaration

- Compute: single Apple-MPS device. No proprietary or large-scale compute.
- Data: two public corpora (donateacry-corpus, Donate-A-Cry-Augmented). No private, IRB-restricted, or licensed data.
- License: code MIT in supplementary; data follows upstream licenses (donateacry MIT, Kaggle redistribution inherits).

### Ethics statement

The paper's §6 (Broader Impact and Ethics) is the ethics statement. Highlights:
- "Decision-support, not diagnosis" framing.
- Demographic-representativeness caveat (donateacry is parent-uploaded, English-speaking, smartphone-connected; Kaggle redistribution adds no demographic diversity).
- Cross-source generalization is **not** demonstrated — flagged as a primary limitation, not papered over.
- Fairness-relevant data-hygiene finding (cross-class duplication in the Kaggle redistribution) is itself an exportable contribution.

---

## Sanity-check before clicking submit

- [ ] Tarball compiles end-to-end via Tectonic (verified locally; OpenReview also compiles).
- [ ] 7 pages of main content + 1 page references = 8 pages — within Proceedings limit.
- [ ] No identifying text in `main.tex`, `main.bbl`, or figure captions. Identity-leak grep run: clean.
- [ ] Author block reads "Anonymous Authors / Anonymous Institution / anonymous@anon".
- [ ] Code/data paragraph says "repository URL withheld for double-blind review" (not the GitHub URL).
- [ ] All 4 figures and 3 tables referenced and visible.
- [ ] Bibliography compiled correctly (`main.bbl` in tarball, no missing references).
- [ ] **Repo visibility decided** (private during review OR anonymous-4open-science mirror prepared).

---

## After submission

1. OpenReview assigns the paper an ID and an Area Chair.
2. Reviewing is single-pass (no rebuttal in some ML4H years; check the year's specific instructions when the CFP opens).
3. Decisions typically land ~6-8 weeks after the deadline.
4. If accepted to Proceedings: prepare PMLR camera-ready (de-anonymize, add author block + Code/data URL + Zenodo DOI, update jmlr.cls to non-anonymous mode). Update arXiv version to the camera-ready.
5. If accepted to Findings: shorten to ~4 pages, drop appendix material, keep the headline result + the data-hygiene finding.
6. If rejected: the reviews are usable. Natural next venues: ICASSP / Interspeech (audio-ML), or NeurIPS Datasets & Benchmarks for the data-hygiene angle as its own paper.

---

## Things explicitly not in this submission (deliberate scope choices)

- No genuine cross-source eval. Acknowledged in §6 limitations. The hardest reviewer ask but the paper is honest about it.
- No DDPM in AST embedding space. Named as the obvious follow-up.
- No clinical validation. Out of scope for a single-author-team applied ML paper.
- No PANNs CNN14 comparison (AST is the chosen backbone proxy; PANNs would tell a similar story).
- No demographic-subgroup audit beyond corpus-level representativeness statements — donateacry does not ship demographic labels, so any subgroup audit would require manual annotation we did not perform.

---

## Timing

ML4H 2026 CFP timing follows the prior years' pattern:
- CFP open: ~July 2026.
- Paper deadline: ~September 2026.
- Symposium: ~December 2026 (co-located with NeurIPS).

We are submission-ready **now** (May 2026); the bundle, abstract, keywords, and ethics statement are all locked in. When the OpenReview venue goes live, the submission is a copy-paste-and-upload operation.

Until then: hold the bundle, do not push to arXiv (a public arXiv version would compromise the double-blind property unless we time it to land after the ML4H deadline — easier to just hold).

---

## File-by-file summary

| File on disk | What it is |
|---|---|
| `report/infant-cry-augmentation-ml4h.tar.gz` | **Upload this to OpenReview.** Self-contained source bundle. |
| `report/ml4h_bundle/main.tex` | Anonymized LaTeX source (two-column 10pt PMLR/JMLR). |
| `report/ml4h_bundle/main.bib` | Bibliography database. |
| `report/ml4h_bundle/main.bbl` | Pre-built bibliography (IS in the tarball). |
| `report/ml4h_bundle/main.pdf` | Local PDF preview (NOT in the tarball; for visual checks). |
| `report/ml4h_bundle/jmlr.cls` | PMLR/JMLR class file (IN tarball — guarantees portability). |
| `report/ml4h_bundle/jmlrutils.sty` | PMLR/JMLR utility macros (IN tarball). |
| `report/ml4h_bundle/figures/` | 12 PNGs; only 4 are referenced by main.tex. |
| `report/ml4h_submission.md` | This file — metadata + checklist. |
