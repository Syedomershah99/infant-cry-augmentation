# arXiv submission metadata (v2 — with AST backbone comparison)

Submission portal: <https://arxiv.org/submit>

---

## Source bundle to upload

`report/infant-cry-augmentation-arxiv.tar.gz` (1.7 MB)

The tarball contains:

| File | Purpose |
|---|---|
| `main.tex` | Main paper source (with `\usepackage[preprint]{tmlr}` so authors are visible) |
| `main.bib` | BibTeX bibliography |
| `main.bbl` | Pre-built bibliography (so arXiv does NOT need to run BibTeX) |
| `tmlr.sty` | TMLR LaTeX class file |
| `tmlr.bst` | TMLR bibliography style |
| `fancyhdr.sty` | Header style required by `tmlr.sty` |
| `math_commands.tex` | Optional math macros (referenced by tmlr.sty) |
| `figures/` (12 PNGs) | All figures in the paper |

No PDF in the tarball — **arXiv regenerates the PDF from source.**

---

## Step 1 — arXiv submission form: Identify

- **Submission type**: New submission
- **License**: CC BY 4.0 (Creative Commons Attribution 4.0)

---

## Step 2 — Add files

Upload `infant-cry-augmentation-arxiv.tar.gz` to the "Add files" panel. After upload, arXiv processes the tarball. If asked which file is the main one, select `main.tex`.

If arXiv reports "missing fancyhdr.sty" or similar, all four style files are inside the tarball — re-uploading usually fixes it.

---

## Step 3 — Metadata

### Title

```
Class-Conditional Generative Augmentation for Rare-Class Infant Cry Classification
```

### Authors

```
Syed Omer Shah, Collin Murphy
```

### Affiliation

```
University at Buffalo, Department of Computer Science and Engineering, Buffalo, NY 14260, USA
```

### Abstract (verbatim)

```
Infant cry classification is a clinically-adjacent task in which the rarest classes (such as pain) are also the most safety-critical. Public corpora are heavily imbalanced; in the donateacry-corpus, hungry accounts for 84% of clips and burping for 1.8%. A naive log-mel CNN achieves 84% aggregate accuracy on this corpus by collapsing onto the majority class, missing every safety-critical case. We study a small class-conditional denoising diffusion model (DDPM) as a targeted fairness intervention for the rare classes, combined with a 3-way optimizer-recipe ablation (class-weighted cross-entropy, balanced sampler, focal loss). Integrating a second public corpus surfaces a fairness-relevant data-quality issue: many redistributed rare-class clips are byte-identical to clips of other classes, which we deduplicate at the content level. We then report a controlled 4 x 3 x 3-seed augmentation matrix (36 cells), a noise-robustness sweep, a synth-to-real ratio sweep, and a backbone comparison against a frozen AudioSet-pretrained AST linear probe. The best cell, generative + weighted_ce, attains macro-F1 = 0.258 (vs. 0.194 baseline; +33%) and burping recall = 0.667 (vs. 0.000 across all 18 non-generative cells) while preserving 81.6% accuracy. Under additive noise the generative arm holds 76.8% accuracy at 10 dB SNR where the baseline drops to 54.6%. The ratio sweep exposes a non-monotonic threshold effect: 1x and 2x synthetic ratios behave like the baseline; 3x crosses an argmax threshold and burping recall jumps from 0 to 0.667. The AST backbone comparison shows that pretraining recovers belly_pain on 1 of 3 seeds (the small CNN never does), but a Gaussian-proxy generative arm in AST embedding space does not recover burping, indicating that pretraining and in-mel-space DDPM are doing orthogonal work. We release the code, manifests, datasheet, model card, and per-cell artifacts.
```

### Comments

```
9 pages, 4 figures, 5 tables. Code, manifests, datasheet, model card, and per-cell artifacts: https://github.com/Syedomershah99/infant-cry-augmentation
```

### Primary subject class

```
cs.LG (Machine Learning)
```

**Why cs.LG as primary**: the author's existing arXiv endorsement is in `cs.LG` (ML). arXiv endorsements are category-specific and do not transfer. An earlier submission with `cs.SD` as primary was rejected with arXiv's generic "needs more review" template, which is what moderators apply when no valid endorser exists for the chosen primary. The paper itself is defensibly a machine-learning paper (class-imbalance interventions, generative augmentation, fairness audits applied to audio data) — `cs.LG` is the correct primary for the endorsement we hold.

### Secondary (cross-list) subject classes

```
cs.SD (Sound)
eess.AS (Audio and Speech Processing)
cs.CY (Computers and Society)
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

## Step 4 — Endorsement

You said you have endorsement in `cs.SD` or `cs.LG`. arXiv will route the submission to that endorser automatically for your first submission in those categories.

---

## After arXiv announcement

1. arXiv assigns an ID (e.g., `arXiv:2605.XXXXX`) and posts within ~1 business day after endorsement clears.
2. Update GitHub README with the arXiv URL.
3. (Optional) Update the LaTeX source `Code and data` paragraph to include the arXiv ID.
4. Get a Zenodo DOI for the repo (separate from arXiv) — GitHub → Settings → Integrations → Zenodo.
5. Once both the arXiv URL and Zenodo DOI exist, cite them in any subsequent submission (TMLR is already set up to consume them in the camera-ready phase).

---

## Sanity check before submitting

- [ ] Tarball compiles end-to-end (verified locally with `tectonic main.tex`).
- [ ] `main.bbl` is in the tarball (so arXiv skips BibTeX).
- [ ] Author block shows both authors with emails + UB affiliation.
- [ ] No "Under review as submission to TMLR" header (verified: `\usepackage[preprint]{tmlr}` removes it).
- [ ] No "Anonymous authors" or "URL withheld for double-blind review" remaining (identity-scan passed).
- [ ] Code/data paragraph and Contributions paragraph contain the public GitHub URL.
- [ ] License: CC BY 4.0 chosen (consistent with the repo's MIT code license).
- [ ] All figures resolve in `figures/` (12 PNGs).
- [ ] Page count: 9 pages including the new Section 5.5 (AST backbone comparison).
- [ ] Repository tag `v1.3-arxiv` matches the submitted source.

---

## File-by-file summary

| File on your disk | What it is |
|---|---|
| `report/infant-cry-augmentation-arxiv.tar.gz` | **Upload this to arXiv.** |
| `report/arxiv_bundle/main.tex` | The single LaTeX source (de-anonymized, preprint form). |
| `report/arxiv_bundle/main.bib` | Bibliography database. |
| `report/arxiv_bundle/main.pdf` | Local PDF preview (NOT in the tarball; for visual checks). |
| `report/arxiv_bundle/main.bbl` | Pre-built bibliography (IS in the tarball). |
| `report/arxiv_submission.md` | This file — metadata + checklist. |

You only need the tarball + the metadata in this file. Everything else is local working copy.
