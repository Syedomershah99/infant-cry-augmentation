# TMLR submission metadata

Submission portal: <https://openreview.net/group?id=TMLR>

Source tarball to upload (compiled with the official TMLR style):
`report/infant-cry-augmentation-tmlr.tar.gz` (1.7 MB, 21 files)

The tarball contains the anonymized LaTeX source (`main.tex`), the pre-built bibliography (`main.bbl` plus `main.bib`), the TMLR style files (`tmlr.sty`, `fancyhdr.sty`, `tmlr.bst`, `math_commands.tex`), and 12 figure PNGs. No PDF in the tarball — OpenReview compiles the source.

---

## Before you submit

- [ ] **Create OpenReview profile** for each author (real names + affiliations). Profiles are hidden from reviewers until decision but TMLR requires complete profiles for editorial routing.
- [ ] Both author profiles linked to the submission in OpenReview.
- [ ] Decide on the supplementary archive (the public repo). Do NOT link to it in the paper text; the paper's "Code and data" paragraph already says "URL withheld for double-blind review."

## OpenReview submission form

### Title

```
Class-Conditional Generative Augmentation for Rare-Class Infant Cry Classification
```

### Abstract

```
Infant cry classification is a clinically-adjacent task in which the rarest classes (such as pain) are also the most safety-critical. Public corpora are heavily imbalanced; in the donateacry-corpus, hungry accounts for 84% of clips and burping for 1.8%. A naive log-mel CNN achieves 84% aggregate accuracy on this corpus by collapsing onto the majority class, missing every safety-critical case. We study a small class-conditional denoising diffusion model (DDPM) as a targeted fairness intervention for the rare classes, combined with a 3-way optimizer-recipe ablation (class-weighted cross-entropy, balanced sampler, focal loss). Integrating a second public corpus surfaces a fairness-relevant data-quality issue: many redistributed rare-class clips are byte-identical to clips of other classes, which we deduplicate at the content level. We then report a controlled 4 x 3 x 3-seed augmentation matrix (36 cells), a noise-robustness sweep, a synth-to-real ratio sweep, and a backbone comparison against a frozen AudioSet-pretrained AST linear probe. The best cell, generative + weighted_ce, attains macro-F1 = 0.258 (vs. 0.194 baseline; +33%) and burping recall = 0.667 (vs. 0.000 across all 18 non-generative cells) while preserving 81.6% accuracy. Under additive noise the generative arm holds 76.8% accuracy at 10 dB SNR where the baseline drops to 54.6%. The ratio sweep exposes a non-monotonic threshold effect: 1x and 2x synthetic ratios behave like the baseline; 3x crosses an argmax threshold and burping recall jumps from 0 to 0.667. The AST backbone comparison shows that pretraining recovers belly_pain on 1 of 3 seeds (the small CNN never does), but a Gaussian-proxy generative arm in AST embedding space does not recover burping, indicating that pretraining and in-mel-space DDPM are doing orthogonal work. We release the code, manifests, datasheet, model card, and per-cell artifacts in the supplementary.
```

### TL;DR / Short summary

```
A class-conditional DDPM in mel space, used as a targeted rare-class fairness intervention, recovers burping (recall 0 -> 0.667) and gains +33% macro-F1 over the no-augmentation baseline on donateacry-corpus, with no accuracy penalty. A pretrained AST backbone is complementary (it recovers belly_pain that the DDPM does not). We surface a fairness-relevant cross-class label-noise issue in the most widely redistributed Kaggle infant-cry corpus.
```

### Authors

Add via OpenReview profiles. Profile names and affiliations must be filled but are hidden from reviewers until decision.

### Keywords

```
class imbalance, fairness, data augmentation, denoising diffusion models, audio classification, infant cry, clinical machine learning, fairness audits
```

### Primary area (TMLR uses informal subject areas — pick the closest)

```
Applications: Health / Audio
```

Secondary keywords: `Generative Models`, `Fairness`, `Class Imbalance`, `Data Augmentation`.

### Software, hardware, and data declaration

- Compute: single Apple-MPS device. No proprietary or large-scale compute used.
- Data: two public corpora (donateacry-corpus, Donate-A-Cry-Augmented). No private or licensed data.
- License: code is MIT in the supplementary; data follows upstream licenses (donateacry is MIT-corpus, Kaggle redistribution inherits).

### Conflict declaration

None known. The work is independent academic research.

---

## Sanity-check before clicking submit

- [ ] Tarball compiles end-to-end on the official TMLR LaTeX system (verified locally with Tectonic; OpenReview also compiles).
- [ ] No identifying text in `main.tex`, `main.bbl`, or figure captions. (Identity-leak grep run: clean.)
- [ ] Footer reads "Under review as submission to TMLR" (default `\usepackage{tmlr}` behavior, no `[accepted]` or `[preprint]` option).
- [ ] Author block reads "Anonymous authors / Paper under double-blind review" (auto by tmlr.sty).
- [ ] Code/data paragraph says "URL withheld for double-blind review" (not the GitHub URL).
- [ ] Page count: 9 pages including the new Section 5.5 backbone comparison (no page limit at TMLR).
- [ ] All 5 tables and 4 figures referenced and visible.
- [ ] Bibliography compiled correctly (`main.bbl` in tarball).

---

## After submission

1. OpenReview assigns the paper an ID and an Action Editor within ~1 week.
2. The Action Editor checks scope and anonymity, then assigns 3 reviewers.
3. First reviews expected within ~4 weeks of AE assignment.
4. Rebuttal / discussion phase is open-ended; allow another ~4--6 weeks.
5. First decision typically lands ~2--3 months after submission.

When the decision arrives, the next action depends on the verdict:
- **Accept / accept with minor revisions:** update camera-ready (use `\usepackage[accepted]{tmlr}`), de-anonymize, add author block + Code/data URL + Zenodo DOI. arXiv version can also be updated to the published version.
- **Major revision:** address reviewer asks; most common is the AST/DDPM-in-embedding-space follow-up the paper already names as future work.
- **Reject:** the reviews are usable. The natural next venue is ML4H @ NeurIPS (workshop) or a Datasets-and-Benchmarks-style spin-off for the data-quality finding.

---

## Things explicitly not in this submission (deliberate scope choices)

- No genuine cross-source eval. Acknowledged in limitations (Section 7). Hardest reviewer ask but the paper is honest about it.
- No DDPM in AST embedding space. Named as the obvious follow-up paper.
- No clinical validation. Out of scope for course-project-sized work.
- No PANNs CNN14 comparison (AST is the chosen backbone proxy; PANNs would tell a similar story).
