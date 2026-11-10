# Project status — 1 November 2026 (evening update)

## Update 11 November 2026: manuscript on Overleaf, Scientific Reports layout

- **Overleaf is connected** (claudeleaf). The manuscript lives in the Overleaf project "ECG Paper - Miriam"
  (owner: Dr. Elgendi). It held an older Mendeley/Scientific Reports paper; its text is backed up in
  `paper/overleaf_backup_2026-11-10_before_replace/`, and its other files were left in the project unused.
- **Layout changed at the supervisor's request:** single column, no "(submitted)" running head. The new source is
  `paper/ieee_manuscript/main_source_scirep.tex` (`wlscirep.cls`, the Scientific Reports template already in the
  project): same text and numbers, sections in Scientific Reports order (Introduction, Results, Discussion,
  Methods), related work folded into the Introduction, Author contributions and Competing interests added as
  placeholders. Build it with `python -X utf8 paper/ieee_manuscript/build_overleaf.py scirep`
  (→ `overleaf_scirep/`, `overleaf_scirep.zip`). The IEEE version (`main_source.tex`) is kept and builds as before.
- **Compiled on Overleaf:** 10 pages, no errors, no undefined references or citations, no overfull boxes.
- Figures are at the project root on Overleaf (the connection cannot create folders); `\graphicspath` finds them
  there and in `figures/` locally.
- The AI-assistance sentence was removed from the acknowledgment at the user's request (both versions).
- Still open: affiliations, funding, corresponding e-mail, author contributions, competing interests.
- The Scientific Reports abstract was shortened from 230 to 199 words (limit 200), with the same numbers and
  claims; the IEEE abstract (230 words, within IEEE's 250) is unchanged.
- **GitHub.** The local repository now tracks https://github.com/itsmiri-eng/ecg-image-gain-sequence-models
  (renamed 11 Nov 2026 from `ecg-image-domain-shift`, which GitHub redirects; branch
  `main`). Its earlier content (the September Mendeley project) is kept in the history and under the tag
  `v1-mendeley-2026-09`. The new README describes the current study. Commits use the GitHub noreply e-mail.
  The Overleaf backup folder (supervisor's e-mail and project id) is kept out of Git. Miriam pushes with
  GitHub Desktop from her clone in `OneDrive/Documents/GitHub/ecg-image-domain-shift` (folder name unchanged). The repository is
  public, while the manuscript says code is released on publication; the authors decide whether it stays public.

## Update 10 November 2026: manuscript, figures and reproducibility package

- **IEEE manuscript draft** (target IEEE JBHI), in `paper/ieee_manuscript/`:
  - `main_source.tex`: the editable source.
  - `overleaf.zip`: upload to Overleaf with New Project → Upload Project, then compile with pdfLaTeX + BibTeX.
  - The draft was **not compiled on this PC**, because no LaTeX is installed. It passes static checks
    (`build_overleaf.py`): environments, braces, references, citation keys, figure files and table column counts.
  - Tables come from `src/make_paper_tables.py` and figures from `src/make_paper_figures.py`, so nothing is
    typed by hand.
  - `references.bib`: 37 entries built from Crossref/DataCite records; every DOI resolves.
  - Affiliations are left as placeholders, at the user's request.
  - The acknowledgment discloses AI assistance, as IEEE policy requires. The authors must review that wording.
- **New analysis.** The PTB-XL fixed − adaptive comparison now uses the same hierarchical seed × patient bootstrap
  as the external H2 (`results/fixed_gain/retrained_comparison/hierarchical_test.csv`). Seeds 42–46, Holm over 8:
  - CNN-only +0.030 macro / +0.073 HYP;
  - temporal conv +0.016 / +0.043;
  - LSTM +0.012 (p=0.028) / +0.014 (p=0.11);
  - ViT +0.006 / +0.010 (n.s.).
- **Calibration and threshold tables** now cover every run, including seeds 45/46 and the wide head.
  - Mean test ECE: CNN-only 0.026, LSTM 0.026, temporal conv 0.033, ViT 0.046.
- **Corrections found while writing.**
  - The CNN-LSTM is **unidirectional** (1 layer, 32 hidden units). Model cards and the handover file said
    bidirectional; both are fixed.
  - Seed 42 of **CNN-only** was also Mac-CPU-trained.
- **Supervisor summary:** `docs/supervisor_summary_2026-11-10.md`, plus a shareable Claude Doc.
- **Reproducibility package.**
  - A local Git repository was created, with author Miriam Istanboli. It has **not been pushed**; she pushes it
    herself.
  - `.gitattributes` stores files byte-for-byte, so the hash-bound records stay valid on any computer.
  - `.gitignore` excludes personal and third-party files, Teams drafts, CLAUDE.md, external data and checkpoints.
  - About 470 MB is tracked; the largest file is 16 MB.
  - `docs/REPRODUCIBILITY.md` now covers the external test.
  - Code licence: still to be chosen by the authors.

## Update 7 November 2026: evidence complete

- **Training.** All 21 amendment runs are complete: seeds 45/46 of every network on both gain settings, and
  the capacity-matched head for seeds 42–46.
  - The external test ran once per run and dataset: 52 runs × Georgia (6,047) and China (9,004).
  - Tests: 239 passed.
  - Full results and interpretation are in `paper/EXPERIMENT_NOTES.md`, under "Results update, 7 November 2026".
- **PTB-XL headline (5 seeds).** CNN-only 0.830 < capacity-matched CNN 0.865 < {CNN-LSTM 0.877, temporal conv 0.879,
  ViT 0.886}.
  - Relating information along the row adds 0.011–0.020 beyond head capacity (Holm-significant).
  - The three sequence heads are not separable on PTB-XL test.
- **External headline.**
  - The ordering CNN < wide CNN < {LSTM, ViT} replicates in the USA and China datasets. The temporal-conv
    head is unstable across datasets.
  - **Fixed-gain rendering improves every architecture on both unseen datasets**, most for local models. This
    confirms the main contribution.
  - The signal model remains best: the image representation costs about 0.03–0.05 externally.
  - All models drop substantially out of distribution.
- **Remaining work.** No heavy compute is needed:
  1. reproducibility package finalization (git repository for the user to push, Zenodo archive);
  2. figures for the external results;
  3. the supervisor meeting;
  4. the manuscript.
  - Optional, and only if requested: a pretrained ResNet-18 reference (a 45 MB weight download needs the user's OK).

## Update 5 November 2026

- **Paused runs lost.** The 5 runs paused on 3 Nov were closed before they could resume. They were at epoch 15
  of 20 and had no test inference. They were moved aside, unedited:
  [abandoned_runs/2026-11-03_paused_then_closed](../experiments/abandoned_runs/2026-11-03_paused_then_closed/README.md).
- **Restart.** All 21 amendment runs restarted at 09:32 UTC in 4 lanes, capped at 8 of 16 logical CPUs and
  idle priority (`experiments/run_lane.ps1`).
- **Label mapping built and external plan FROZEN** (10:34 UTC; `experiments/external/plan_freeze.json`):
  - Age and sex align on all 21,799 PTB-XL headers, with 0 mismatches.
  - 25 of 50 SNOMED codes are assigned.
  - NORM has no external label, because the organisers coded NORM as sinus rhythm.
  - T-wave abnormal (STTC share 0.88) falls just below the 90% rule.
  - No external ECG header or signal had been parsed before the freeze.
- **External pipeline written:** `src/external_test.py`, with steps inventory → render → infer → analyze.
  - 7 synthetic tests pass.
  - Reload check on PTB-XL validation: all 31 completed runs reproduce their saved probabilities. Networks match
    to ≤7e-7, LDA ≤1.5e-5, SVM 3e-16 and the signal model ≤2.2e-6. The Mac-CPU ViT is at 4.3e-4, which is known.
  - The pipeline runs when the main download completes.
- `src/compare_ptbxl_all.py` now includes `cnn_pool_wide`.

## Update 3 November 2026 (morning)

- **Protocol amendment frozen before any new run** ([protocol_amendment_2026-11-03.md](protocol_amendment_2026-11-03.md)).
  - Seeds 45/46 for all four networks, on adaptive and fixed gain.
  - A capacity-matched CNN-only head, `cnn_pool_wide`: per-column MLP, no column mixing, 17,065 parameters
    against 17,161 for temporal conv.
  - A hierarchical (seed × patient) bootstrap with Holm correction for the 10 head pairs.
- **21 GPU runs started at 10:33 UTC in five parallel lanes:** 8 adaptive, 8 fixed-gain and 5 wide-head runs.
  - Training code is unchanged from seeds 43/44.
  - No data-loading change was made. It would have changed the hash-bound trainer between seeds; running more
    lanes at once was used instead.
  - Logs: `experiments/logs/run_queue_pc_20261103_*` and `experiments/logs/<run_id>.log`.
- **External-test analysis plan drafted:** [external_test_plan.md](external_test_plan.md).
  - Primary set: Georgia (CinC 2021). Secondary: Chapman-Shaoxing + Ningbo.
  - Labels come through the published PTB-XL+ SNOMED mapping, not a hand-made one.
  - Nothing has been downloaded yet. Downloads need the user's OK.
- **Reproducibility package started:**
  - [REPRODUCIBILITY.md](REPRODUCIBILITY.md), [data_statement.md](data_statement.md) and
    [model_cards.md](model_cards.md) (`src/make_model_cards.py`);
  - `environment/requirements-lock-pc-2026-11-03.txt`;
  - `.gitignore` additions.
- **Observation for the limitations section.** Many runs select their checkpoint at epoch 19–20 of 20,
  especially CNN-only (all seeds) and the fixed-gain runs. With a cosine schedule that ends at epoch 20, this
  is expected. It still means the 20-epoch budget may limit some models, CNN-only most. It is reported, not
  tuned: the epoch budget is prespecified.
- **Tests:** 231 passed, 3 skipped. This includes 4 new tests for the wide head.
- **External data download approved by the user and running.**
  - Command: `src/download_external.py`, which checks every file against PhysioNet SHA-256 and can be resumed.
  - Contents: Georgia, Chapman-Shaoxing and Ningbo, into `data/external/`. Log:
    `experiments/logs/download_external_20261103b.out.log`.
- **Mapping problem.** The PTB-XL+ mapping file uses OMOP concept IDs, not SNOMED CT codes, so it cannot label
  the Challenge headers. A replacement (the Challenge organisers' own PTB-XL → SNOMED conversion) is in
  §3 of the external-test plan. It needs one more small download (≈15 MB of header text files), and the user
  has to approve that.

## Update 2 November 2026 (Windows PC, RTX 5070 Ti)

- **All 13 planned GPU runs finished** (CNN-only seeds 42–44; CNN-LSTM, temporal conv, ViT seeds 43/44; four fixed-gain seed-42 runs). A Windows path bug stopped each one after its single test inference and before its log row; records were rebuilt from saved files without repeating inference ([record_recovery](../experiments/record_recovery/README.md)).
- **Windows portability fixed** ([windows_portability_2026-11-02.md](windows_portability_2026-11-02.md)): tests 218 passed, 3 skipped; the three v1 seed-42 runs pass full re-verification on the PC.
- **Three-seed results** (test macro AUROC, mean ± SD, seed 42 = v1 run, pooling justified by the trainer equivalence check): CNN-only 0.829 ± 0.005, CNN-LSTM 0.879 ± 0.008, temporal conv 0.879 ± 0.001, ViT 0.888 ± 0.002. CNN-LSTM and temporal conv are **tied**; the seed-42 "+0.01 per step" does not hold for that step.
- **PCA + linear SVM baseline** (new, own frozen config): test macro AUROC 0.793 [0.782, 0.806], indistinguishable from PCA + LDA (0.794). Added as the most likely reading of "vsm" in the user's notes; see the roadmap.
- **Fixed-gain test-fold transfer done** (one inference per model; [results/fixed_gain/transfer_test](../results/fixed_gain/transfer_test/summary.md)): networks lose 0.002–0.008 macro AUROC on re-drawn test pages, LDA 0.043; HYP sensitivity rises on re-drawn pages, as on validation. CNN-only was added to the transfer as a disclosed addition.
- **Signal-domain reference done** (roadmap item 2; protocol `experiments/configs/ptbxl_v2_signal_resnet1d.json`, frozen before training): Wang-style 1D ResNet on the raw 100 Hz waveforms of the same 21,388 ECGs, same folds, order, loop, hyperparameters and test policy. Test macro AUROC 0.929 / 0.928 / 0.928 (seeds 42/43/44; mean 0.928 ± 0.001), in line with published PTB-XL signal benchmarks (≈0.93). Against the best image model (hybrid ViT, seed 42) the paired difference is +0.039 [0.032, 0.046] macro AUROC: the cost of the paper-image representation at this model size. Waveform cache: `data/processed/ptbxl_signals100_v1` (every source file checked against PTB-XL's SHA256SUMS). The signal model has ≈0.5 M parameters versus ≤24k for the image heads, so the gap mixes representation and capacity.
- **Secondary analyses done** (roadmap items 6-7; plan frozen first in [secondary_analyses_protocol.md](secondary_analyses_protocol.md)):
  - *Calibration* ([results/secondary/calibration_thresholds](../results/secondary/calibration_thresholds/summary.md)): every model is reasonably calibrated (test macro ECE 0.02-0.05); the hybrid ViT is the least calibrated (0.04-0.05) and over-predicts HYP.
  - *Validation-chosen thresholds*: test macro F1 rises for every model (e.g. CNN-only 0.50 → 0.61, LDA 0.43 → 0.57, ViT 0.67 → 0.69, signal 0.73 → 0.75), so the fixed 0.5 rule mainly penalizes the weaker models. 0.5 stays primary.
  - *Saliency* ([results/secondary/saliency](../results/secondary/saliency/summary.md)): share of the HYP output's integrated-gradient attribution that falls on the 1 mV calibration pulse (3.6% of each row), HYP-positive validation ECGs, seed 42. Adaptive-gain models: CNN-only 5.3%, temporal conv 7.6%, CNN-LSTM 11.4%, **hybrid ViT 24.7% [23.8, 25.5]**; the same architectures retrained on fixed-gain pages: 1.4-3.2%. The pulse height encodes page gain, so the adaptive-gain models, above all the ViT, read the gain from the pulse (recomputed in full float32: identical to 4 decimals).
  - *Pulse-swap intervention* (protocol D; [results/secondary/pulse_swap](../results/secondary/pulse_swap/summary.md)): changing **only** the 40 pulse columns of each validation page, nothing else. Putting the short 2.5 mm/mV pulse on pages drawn at 5-10 mm/mV raises the adaptive-gain models' mean HYP probability by **+0.49** [0.48, 0.51] (ViT), +0.31 (CNN-LSTM), +0.15 (temporal conv), +0.04 (CNN-only); putting the tall 10 mm/mV pulse on 2.5 mm/mV pages lowers it by −0.36, −0.30, −0.19, −0.05. Validation HYP AUROC of the ViT on 5-10 mm/mV pages falls from 0.836 to 0.700 with the short pulse. A same-gain copy of the block changes nothing (pulse blocks are pixel-identical within a gain). Fixed-gain retrained models move by at most 0.014. This shows the adaptive-gain networks' HYP output depends strongly on the pulse. **Correction (3 Nov):** an earlier version of this line called that causal evidence of a shortcut. That overstated it: reading the pulse as a voltage scale is what a cardiologist does, and it predicts the same direction. A post-hoc split ([decomposition_posthoc](../results/secondary/pulse_swap/decomposition_posthoc.md), specified after these results) compares the swap with the consistent re-render of the transfer test, where waveform and pulse change together and true voltage is unchanged: the renderer's gain prior moves HYP probability by only +0.03 to +0.08 in every model, and the remainder (≈ calibration reading) is +0.46 ViT, +0.23 CNN-LSTM, +0.11 temporal conv, +0.00 CNN-only. Approximate (re-rendering also shrinks the waveform). Checks: unchanged pages reproduce saved validation probabilities to ≤4e-6, except the Mac-trained ViT (4.3e-4, the CPU→GPU inference drift already measured by the PC setup check). Fixed-gain profiles also show regular peaks every 20 columns (5 mm grid lines), and CNN-LSTM attends to the left margin in both conditions; both are noted, not interpreted yet.
- **Fixed vs adaptive gain on matched hardware done** (fixed-gain seeds 43/44 trained on the PC GPU, 3 Nov; [results/fixed_gain/retrained_comparison](../results/fixed_gain/retrained_comparison/summary.md), paired patient bootstrap per seed, validation primary). Fixed − adaptive validation macro AUROC, seeds 43 / 44: CNN-only +0.034 / +0.031, temporal conv +0.015 / +0.018 (all intervals above zero), ViT +0.004 / +0.008, CNN-LSTM +0.005 / −0.003 (intervals include zero except ViT seed 44). HYP AUROC follows the same pattern (CNN-only +0.07 / +0.10, temporal conv +0.08 / +0.04; ViT and CNN-LSTM ≈ 0). No model is clearly worse with fixed gain (only ViT seed 43 on test, −0.007 [−0.013, −0.001], not repeated on validation or seed 44). Protocol hypotheses: H2a (HYP maintained or improved) holds for every network.
- **Emerging interpretation (post-hoc, to be tested):** the models that read the pulse as a voltage scale most (ViT, CNN-LSTM, which see a whole row at once) gain least from fixed gain; the models that cannot relate the pulse to the distant waveform (CNN-only, temporal conv with a 13-step receptive field) gain most. Fixed gain gives every model the voltage directly, which matters most for those that cannot calibrate. The renderer's gain prior itself is a small effect.
- **Next steps:** [ROADMAP_Q1.md](ROADMAP_Q1.md) replaces the list at the bottom of this page.

This page states what exists and what has actually been run. Earlier versions are preserved in [project_status_before_resume.md](../experiments/reviews/2026-11-01/project_status_before_resume.md) and [project_status_codex_1nov_superseded.md](../experiments/reviews/2026-11-01/project_status_codex_1nov_superseded.md).

## What the project is

A comparison of progressively more complex classifiers for **12-lead ECG images**. The main benchmark is a custom clean rendering of every PTB-XL v1.0.3 record into a paper-style page image (21,799 pages), classified into five overlapping diagnostic superclasses (NORM, MI, STTC, CD, HYP). A smaller, earlier four-class Mendeley paper-ECG image set (Normal, MI, Abnormal, History of MI) is kept as a separate preliminary study; its results must never be ranked against PTB-XL results.

| Item | PTB-XL image benchmark (main) | Mendeley image set (preliminary) |
|---|---|---|
| Records | 21,388 eligible ECGs (411 with no diagnostic superclass excluded) | 491 unique images after 437 exact copies removed (928 files) |
| Patients | 18,869 in source; patient IDs exist | Printed report IDs only; not verified patient identities |
| Task | 5 overlapping binary labels (multilabel) | 4 mutually exclusive classes |
| Model input | 12 × 120 × 1120 lead-row image (inverted, 2×2 max-pooled) | 1 × 128 × 256 grayscale crop |
| Split | Official PTB-XL patient folds: 1–8 train (17,084), 9 validation (2,146), 10 test (2,158); no patient overlap | Grouped split 295 / 98 / 98 |
| Split file | `data/derived/ptbxl_images_v1/splits/` and the cache manifest | `data/splits/fixed_split.csv` |

## Data validation (done before this session, re-checked 1 November)

The 1 November audit ([dataset_audit_resume.md](../experiments/reviews/2026-11-01/dataset_audit_resume.md)) passed: all 21,799 source pages and all 21,388 cached model inputs match their checksums; no exact duplicate decoded images, including across partitions; no patient in more than one fold; normalization uses training pixels only; split reconstruction matches the official folds. Mendeley exact duplicates were removed and group overlaps handled; Mendeley patient identity and clinical labels remain unverifiable. Label quality: PTB-XL labels are the dataset's own SCP-code annotations mapped to superclasses (all present diagnostic codes, including likelihood 0); no independent clinical adjudication was done.

**Test-set exposure.** The PTB-XL test fold (fold 10) was first evaluated for the CNN–LSTM in September. Every later model was specified with that knowledge, so all PTB-XL model comparisons are exploratory. No threshold, hyperparameter or checkpoint has been chosen using test data; every run opens the test fold once, after its checkpoint is frozen.

## Completed and verified runs

| # | Run | Data | Status | Test headline |
|---|---|---|---|---|
| 1 | `lda_v1_seed42` | Mendeley | Complete, verified | Accuracy 0.6224, macro F1 0.5637 |
| 2 | `cnn_v1_seed42` | Mendeley | Complete, verified | Accuracy 0.7551, macro F1 0.5621 |
| 3 | `cnn_lstm_image_v1_seed42` | Mendeley | Complete, verified | Accuracy 0.7143, macro F1 0.6020 |
| 4 | `ptbxl_cnn_lstm_image_v1_seed42` | PTB-XL | Complete, verified | Macro AUROC 0.8698, macro F1 0.6131 |
| 5 | `ptbxl_spatiotemporal_image_v1_seed42` | PTB-XL | Complete, verified | Macro AUROC 0.8803, macro F1 0.6409 |
| 6 | `ptbxl_hybrid_vit_image_v1_seed42` | PTB-XL | **Completed 1 Nov 14:55 UTC; verification passed** | Macro AUROC 0.8902 [0.8809–0.9000], macro F1 0.6717 |
| 7 | `ptbxl_v2_lda_seed42` (PCA + LDA, grid ≤256) | PTB-XL | **Completed 1 Nov this session** | Macro AUROC 0.7883 [0.7757–0.8013], macro F1 0.4034 |
| 7b | `ptbxl_v2_lda_wide_seed42` (PCA + LDA, grid 256–1024; disclosed follow-up) | PTB-XL | **Completed 1 Nov this session** | Macro AUROC 0.7939 [0.7822–0.8065], macro F1 0.4269 |

The master PTB-XL table, per-label sensitivities and paired bootstrap differences are in [results/comparisons/ptbxl_master/master_table.md](../results/comparisons/ptbxl_master/master_table.md); regenerate with `python -m src.compare_ptbxl_all`.

## Moving to an NVIDIA PC (decided 1 November, evening)

The MacBook Air (CPU only, 30 W charger) was too slow and overheating, so the remaining runs move to the user's Windows PC with an NVIDIA GPU. Step-by-step guide, including what changed in the code: [docs/MOVE_TO_NVIDIA_PC.md](MOVE_TO_NVIDIA_PC.md).

- **Stopped on the Mac (~21:55 local):** all queues. The paused `ptbxl_v2_cnn_pool_seed43` (epoch 7, no test inference, no log row) was killed and its partial files moved to [experiments/abandoned_runs/2026-11-01_stopped_for_pc_move/](../experiments/abandoned_runs/2026-11-01_stopped_for_pc_move/README.md); it restarts from scratch under the same run ID on the PC. The not-yet-started runs had created no files.
- **Left to finish on the Mac:** `ptbxl_v2_cnn_pool_seed42` (CPU, epoch 16 of 20 at the time). If it is not complete when the folder is copied, the PC queue trains it instead.
- **Queue for the PC (one GPU lane):** cnn_pool 42/43/44; hybrid ViT, temporal conv and CNN–LSTM seeds 43/44; fixed-gain cnn_pool, CNN–LSTM, temporal conv and hybrid ViT seed 42. Check progress with `python -m src.status`.
- **Hardware becomes part of the record.** Seed 42 of every network was trained on the M1 CPU; the remaining seeds and the fixed-gain retrains run on CUDA. Every run stores its device and platform. Expected consequences, to state in the paper:
  - GPU training is generally not bit-reproducible (some CUDA backward kernels, for example adaptive pooling, are nondeterministic). `src.check_pc_setup` measures this on the PC and also measures inference drift of a Mac-trained model; its report goes to `experiments/setup_checks/`. A Mac baseline report already exists there (inference difference 0, training repeat difference 0).
  - In the fixed-gain Q2 comparison, the adaptive-gain seed 42 runs were trained on CPU and the fixed-gain seed 42 runs on GPU, so the pairs also differ in hardware. Seed-to-seed spread (seeds 43/44, GPU) gives the scale of such run-to-run variation. Recommended once the queue is done, because it is cheap on a GPU: fixed-gain seeds 43/44, so that Q2 can also be checked on matched hardware (GPU adaptive 43/44 vs GPU fixed-gain 43/44).

Seed 42 for the three sequence models is the existing v1 run: a check on real training data showed the new shared trainer produces bit-identical weights to each v1 trainer ([trainer_equivalence_check.json](../experiments/reviews/2026-11-01/trainer_equivalence_check.json)).

## What the evidence currently says (PTB-XL, seed 42, exploratory)

- Every learned image model beats the tuned PCA + LDA baseline by a large margin (macro AUROC +0.076 to +0.096).
- Each step CNN–LSTM → temporal convolution → small hybrid ViT raised test macro AUROC by about 0.01, with paired patient-bootstrap intervals above zero (CNN–LSTM 0.8698, TCN 0.8803, ViT 0.8902). Validation shows the same order.
- The ViT is the first model with useful HYP (hypertrophy) sensitivity: 0.531 versus 0.279–0.324; MI sensitivity is 0.498, between CNN–LSTM (0.433) and TCN (0.562).
- The ViT selected epoch 19 of 20 and validation AUROC was still near its peak, so the fixed 20-epoch budget may be limiting it; its train–validation macro AUROC gap is small (0.024), so there is no sign of strong overfitting.
- These are single-seed results. Whether the ~0.01 steps survive seed variation is exactly what the running seed study tests.

## Problems and limitations known now

- **PCA + LDA grid boundary (resolved):** the first grid (16–256) chose its upper edge. A disclosed follow-up grid (256–1024) found an interior validation optimum at 512 components; test macro AUROC moved only from 0.788 to 0.794. The follow-up was specified after the first LDA test result was seen; both runs are reported.
- **Page-gain shortcut for hypertrophy (found this session):** the renderer lowers page gain for high-voltage ECGs, and gain alone predicts HYP with AUROC ≈0.72–0.74. At standard 5–10 mm/mV gain, HYP sensitivity is only 0.02–0.19 at threshold 0.5, although within a single gain level the models still rank HYP well (validation AUROC 0.77–0.84), so the models are not using gain alone. HYP results must be reported stratified by gain, and a gain-controlled rendering experiment is needed before claiming hypertrophy detection from paper ECGs. Details: [results/error_analysis/ptbxl_seed42/summary.md](../results/error_analysis/ptbxl_seed42/summary.md).
- **Capacity confound:** the new CNN-only baseline has 1,705 parameters versus 17,161–23,881 for the sequence heads. It answers "does any sequence head help?", not "is attention better than convolution at equal size".
- **Accelerator:** the Mac GPU (MPS) cannot run the shared encoder's adaptive pooling, so Mac training was CPU-only (≈3 h per run alone, ≈5 h with two in parallel). Remaining runs use an NVIDIA GPU (see above). Runtimes are not a controlled speed benchmark.
- **LSM and the remembered convolution method** ("Portal convolutions") are still undefined in every local source, including the logbook, project description, April report and old manuscripts. Not implemented. One possibility to ask the supervisor about: "Portal" may be a mishearing of "temporal" convolutions, which the project has already run (Run 5). This is a question, not an assumption.
- **No Git repository.** Provenance is kept by SHA-256 hashes and source snapshots in every run folder.
- **Sensitive file (resolved):** a personal ID document was in the Mac copy of the folder. It is not project data, was never transferred to the PC, and is excluded from version control by `.gitignore`.
- **External validity:** clean synthetic renderings only. No real scans/photos have been evaluated. No clinical claims are possible.

## Gain-controlled experiment (started this session)

Protocol: [docs/ptbxl_fixed_gain_protocol.md](ptbxl_fixed_gain_protocol.md). All 21,388 ECGs were re-drawn at one fixed gain (2.5 mm/mV) and verified (cohort identical; unchanged pages pixel-identical). Existing models scored on these pages (validation): networks lose only 0.004–0.010 macro AUROC, LDA loses 0.049; but NORM sensitivity falls and HYP sensitivity rises on re-drawn pages, consistent with the gain layout acting as a hypertrophy prior. LDA retrained on fixed-gain pages: validation macro AUROC 0.790 vs 0.806, with HYP AUROC up (0.853 vs 0.842) and STTC/NORM/MI down. Network retraining is queued. Details: [paper/EXPERIMENT_NOTES.md](../paper/EXPERIMENT_NOTES.md).

## Next steps (in order)

1. Move to the PC and run the queue ([guide](MOVE_TO_NVIDIA_PC.md)); then regenerate the master table (`python -m src.compare_ptbxl_all`), which then reports mean ± SD over seeds 42/43/44.
2. ~~Wider PCA grid for LDA~~ — done (Run 7b).
3. One prespecified ablation: capacity-matched pooled head (same parameter budget as the sequence heads, no sequence mixing), and/or ViT without positional embeddings.
4. ~~First error analysis~~ — done. ~~Gain-controlled rendering~~ — dataset built, transfer and LDA done; network retraining queued on the PC. When done: `python -m src.compare_fixed_gain` (ready; currently LDA pair only: validation −0.016 [−0.026, −0.008]) for paired fixed-vs-adaptive comparisons, then test-fold transfer once per model.
5. Confirm LSM / convolution method with the supervisor; decide on external real-image data.
6. Manuscript: draft in [paper/MANUSCRIPT_DRAFT.md](../paper/MANUSCRIPT_DRAFT.md), notes in [paper/EXPERIMENT_NOTES.md](../paper/EXPERIMENT_NOTES.md).
