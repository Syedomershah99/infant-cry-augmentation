# arXiv submission metadata

Paste these into the arXiv submission form at <https://arxiv.org/submit>.

---

## Source bundle to upload

`report/infant-cry-augmentation-arxiv.tar.gz`

This contains `Report.tex`, `Report.bib`, `Report.bbl` (pre-built bibliography), `nips_2017.sty`, and the `figures/` directory. arXiv will compile from source.

---

## Step 1: Identify

- **Submission type**: New submission
- **Submission license**: CC BY 4.0 (Creative Commons Attribution 4.0)

---

## Step 2: Add files

Upload `infant-cry-augmentation-arxiv.tar.gz`.

After upload, arXiv processes the tarball; the source listing should show `Report.tex` as the main file. If arXiv asks which file is the main one, select `Report.tex`.

---

## Step 3: Metadata

### Title

```
Class-Conditional Generative Augmentation for Rare-Class Infant Cry Classification
```

### Authors

```
Syed Omer Shah, Collin Murphy
```

(format used by arXiv: `Lastname, Firstname; Lastname, Firstname` if asked; comma-separated last names with first names is usually accepted)

### Affiliation (if asked separately)

```
University at Buffalo, Department of Computer Science and Engineering, Buffalo, NY 14260, USA
```

### Abstract

Paste verbatim:

```
Infant cry classification is a clinically-adjacent task in which the rarest classes (such as pain) are also the most safety-critical. Public corpora are heavily imbalanced; in the donateacry-corpus, hungry accounts for 84% of clips and burping for 1.8%. A naive log-mel CNN achieves 84% aggregate accuracy on this corpus by collapsing onto the majority class, missing every safety-critical case. We study a small class-conditional denoising diffusion model (DDPM) as a targeted fairness intervention for the rare classes, combined with a 3-way optimizer-recipe ablation (class-weighted cross-entropy, balanced sampler, focal loss). Integrating a second public corpus surfaces a fairness-relevant data-quality issue: many redistributed rare-class clips are byte-identical to clips of other classes, which we deduplicate at the content level. We then report a controlled 4 x 3 x 3-seed augmentation matrix (36 cells), a noise-robustness sweep, and a synth-to-real ratio sweep. The best cell, generative + weighted_ce, attains macro-F1 = 0.258 (vs. 0.194 baseline; +33%) and burping recall = 0.667 (vs. 0.000 across all 18 non-generative cells) while preserving 81.6% accuracy. Under additive noise the generative arm holds 76.8% accuracy at 10 dB SNR where the baseline drops to 54.6%. The ratio sweep exposes a non-monotonic threshold effect: 1x and 2x synthetic ratios behave like the baseline; 3x crosses an argmax threshold and burping recall jumps from 0 to 0.667. belly_pain recall remains zero across all 36 cells at this data scale, despite a sample-quality probe that classifies 93% of synthetic belly_pain as belly_pain. We release the code, manifests, datasheet, model card, and per-cell artifacts.
```

### Comments

```
10 pages, 4 figures, 4 tables. Code, manifests, datasheet, model card, and per-cell artifacts: https://github.com/Syedomershah99/infant-cry-augmentation
```

### Primary subject class

```
cs.SD (Sound)
```

### Secondary (cross-list) subject classes

```
cs.LG (Machine Learning)
cs.CY (Computers and Society)
eess.AS (Audio and Speech Processing)
```

### Journal reference

Leave blank.

### Report number

Leave blank.

### DOI

Leave blank (will be set if/when published in a venue).

### MSC class / ACM class

Leave blank.

---

## Step 4: Endorsement

You said you already have an endorsement in `cs.SD` or `cs.LG`. arXiv will route the submission to that endorser automatically for the first submission in those categories.

---

## After submission

1. arXiv assigns a paper ID (e.g., `arXiv:2605.XXXXX`).
2. The paper appears on arXiv within ~1 business day after endorsement.
3. Update the GitHub README with the arXiv link.
4. Update the report's `Code and data` paragraph with the arXiv URL once known.
5. Cite this arXiv ID in any subsequent submission (TMLR, ML4H, etc.).

---

## Sanity-check checklist before submitting

- [ ] `Report.pdf` compiled standalone (verified by `tectonic` in `arxiv_bundle/`)
- [ ] `Report.bbl` is in the tarball (so arXiv does not need to rerun `bibtex`)
- [ ] Author block shows both authors with shared affiliation (verified visually)
- [ ] Code URL on first page footer (verified: "Preprint. Code and data: ...")
- [ ] License chosen: CC BY 4.0 (consistent with the repo's MIT code license)
- [ ] No PII in `nips_2017.sty` (no "CSE 4/555" notice; verified)
- [ ] All figures resolve in `figures/` (12 PNGs included)
- [ ] Repository v1.1-arxiv tag pushed and pointing at the matching commit
