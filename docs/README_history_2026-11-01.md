> **Archived copy.** This was the project README until 1 November 2026, kept for the record. It describes the Mac-era workflow and earlier experiments; links were rewritten to work from `docs/`. The current entry points are the repository [README](../README.md) and [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

# ECG image research baselines

## Start here (updated 1 November 2026, evening)

- **Status:** [docs/PROJECT_STATUS.md](PROJECT_STATUS.md) — what is complete, what is running, known problems.
- **Moving to / running on the NVIDIA PC:** [docs/MOVE_TO_NVIDIA_PC.md](MOVE_TO_NVIDIA_PC.md). Progress on any computer: `python -m src.status`.
- **Master PTB-XL comparison:** [results/comparisons/ptbxl_master/master_table.md](../results/comparisons/ptbxl_master/master_table.md)
- **Experiment notes (feed the paper):** [paper/EXPERIMENT_NOTES.md](../paper/EXPERIMENT_NOTES.md); **manuscript draft:** [paper/MANUSCRIPT_DRAFT.md](../paper/MANUSCRIPT_DRAFT.md)

Completed PTB-XL runs (seed 42, test macro AUROC): PCA + LDA 0.788, CNN–LSTM 0.870, CNN + temporal convolution 0.880, CNN + compact hybrid ViT 0.890. The CNN-only baseline, seeds 43/44 of every network and the fixed-gain network retrains run next, on the NVIDIA PC.

### Reproduce (protocol v2: all PTB-XL models, shared trainer)

All commands run from this folder with `/opt/anaconda3/bin/python` (Python 3.12, PyTorch 2.9.1, scikit-learn; see `requirements.txt`). CPU only: the Mac GPU (MPS) cannot run the encoder's adaptive pooling.

```bash
# one network run (models: cnn_pool, cnn_lstm, temporal_conv, hybrid_vit), ~3 h on an M1 CPU
/opt/anaconda3/bin/python -m src.run_ptbxl_experiment --config experiments/configs/ptbxl_v2_cnn_pool.json --seed 42
# PCA + LDA baseline, ~2 min
/opt/anaconda3/bin/python -m src.run_ptbxl_lda --seed 42
# the repeated-seed queues used on 1 November (resumable: finished runs are skipped)
/opt/anaconda3/bin/python -m src.run_queue cnn_pool:42 hybrid_vit:43 temporal_conv:43 cnn_lstm:43 cnn_pool:44
/opt/anaconda3/bin/python -m src.run_queue cnn_pool:43 hybrid_vit:44 temporal_conv:44 cnn_lstm:44
# gain-controlled experiment: build/verify the fixed 2.5 mm/mV cache (~11 min), transfer evaluation, LDA, one network
/opt/anaconda3/bin/python -m src.ptbxl_fixed_gain build --workers 3
/opt/anaconda3/bin/python -m src.ptbxl_fixed_gain verify
/opt/anaconda3/bin/python -m src.evaluate_fixed_gain_transfer
/opt/anaconda3/bin/python -m src.run_ptbxl_lda --seed 42 --config experiments/configs/ptbxl_v2_fg_lda.json
/opt/anaconda3/bin/python -m src.run_queue fg_hybrid_vit:42 fg_temporal_conv:42 fg_cnn_lstm:42 fg_cnn_pool:42
/opt/anaconda3/bin/python -m src.compare_fixed_gain
# rebuild the master table, per-label table, seed summary, paired bootstrap and figure from saved predictions
/opt/anaconda3/bin/python -m src.compare_ptbxl_all
# tests for the v2 code (synthetic data only)
/opt/anaconda3/bin/python -m pytest -q tests/test_ptbxl_v2.py
```

Each run writes `results/ptbxl_v2/<run_id>/` (config, provenance hashes, code snapshot, train/validation/test predictions and metrics, confusion matrices, patient bootstrap, `run.json`) and `models/ptbxl_v2/<run_id>/` (checkpoint, history, learning curve), and appends one row to `experiments/experiment_log.csv`. A run refuses to overwrite an existing run ID and opens the test fold once, after its checkpoint is frozen.

## Historical notes (before 1 November evening)

**Safe integrity checks:** use the new read-only-data audit, writing a new report path. Historical dataset verification commands rewrite timestamps inside hash-bound files and must not be rerun against this frozen dataset.

```bash
/opt/anaconda3/bin/python -B -m src.audit_publication_data --mode full --output experiments/reviews/2026-11-01/dataset_integrity_recheck_01.json
```

The command refuses existing output paths. Both full and metadata-only modes were exercised successfully; metadata-only is explicitly not a full image/source hash audit. Original data and prior verification files remain unchanged. Seven additional audit tests use pytest; run them alongside the existing unittest suite.

The September completion notes below describe historical checkpoints; the current status file is the authority for new work.


This workspace preserves the original ECG images, PTB-XL waveforms, manuscripts, slides, logbook, and historical script. New experiments live in `src/`, `experiments/`, `models/`, and `results/`.

The three earlier Mendeley experiments use **four-class classification of cropped ECG page images**: Normal, MI, Abnormal, and History_MI. Folder labels are used as supplied; they have not been independently clinically adjudicated. Two further completed experiments use separately generated PTB-XL images with five overlapping diagnostic targets. The two datasets and tasks remain separate.

**Fifth experiment completed and verified, 26 September 2026 locally (25 September, 23:17 UTC):** `ptbxl_spatiotemporal_image_v1_seed42` finished 20 epochs and restored validation-selected **epoch 16**. The 17,161-parameter image CNN with a temporal-convolution head uses the same PTB-XL inputs, patient folds and evaluation rules as the CNN–LSTM. Test **macro AUROC is 0.8803 versus 0.8698**, with paired difference **+0.0105** (95% patient-bootstrap interval **+0.0045 to +0.0167**). Macro F1 rises from **0.6131 to 0.6409** at the unchanged 0.5 threshold. See the [frozen protocol](ptbxl_spatiotemporal_protocol.md), [completed report](ptbxl_spatiotemporal_report.md) and [paired comparison](../results/comparisons/ptbxl_spatiotemporal_image_v1_seed42/comparison.md).

This is the **provisional stronger overall PTB-XL candidate**, supported by validation macro AUROC 0.8889 versus 0.8816, rather than a universal best model. Test MI recall improves from **43.27% to 56.18%**, but HYP recall falls from **32.44% to 27.86%**: **189 of 262 HYP positives are missed**. HYP recall also declines on validation. This comparison uses an already-observed test cohort and remains exploratory; patient-bootstrap intervals do not measure training-seed variation or external generalization. Repeated-seed and independent-data studies remain future work. The [research direction](research_direction.md) sets out those next questions.

The fifth run records **one final test inference** after checkpoint selection. Saved-artifact verification passed, all **152 tests pass**, and **164 prior files and the first four central-log rows are preserved**. The log now contains five completed experiments. A [concise supervisor update](teams_update_ptbxl_spatiotemporal.txt) is ready to copy into Teams.

**Earlier completed PTB-XL experiment, 24 September 2026:** [PTB-XL image CNN–LSTM report](ptbxl_cnn_lstm_report.md). Run `ptbxl_cnn_lstm_image_v1_seed42` stopped early after 19 epochs and restored epoch 15, selected using validation macro AUROC. On 2,158 held-out images, test **macro AUROC is 0.8698** (95% patient-bootstrap interval 0.8593–0.8805), macro average precision **0.7035**, and macro F1 **0.6131** at the fixed 0.5 threshold. Exact five-label match accuracy is **52.59%**; individual label-decision accuracy is **84.95%**. These different accuracy definitions must not be ranked against the four-class results below.

In that CNN–LSTM run, recall was weak for HYP (**32.44%**) and MI (**43.27%**). It used one configuration and seed on clean pages rendered from real ECG waveforms, with raster/downsampling loss and variable page gain; real-scan performance remains untested. Its bootstrap interval describes test-patient sampling uncertainty for that fitted model, not training-seed or external-domain variation. Its runner recorded one final test inference after selection. Standalone verification checked all 21,388 saved prediction records and all cached-image hashes without new model inference. At that completion checkpoint, all **112 tests passed**, prior model artifacts and the first three central-log rows were preserved, and the log contained four completed experiments. Those results remain unchanged.

**Dataset completed and verified, 24 September 2026:** [PTB-XL dataset card](ptbxl_image_dataset.md) and [dataset folder guide](../data/derived/ptbxl_images_v1/README.md). All 21,799 native recordings have a full-duration twelve-lead PNG and source/calibration provenance. All images decode and match hashes; no exact rendered-image duplicates or patients crossing partitions were found. Classification uses 21,388 records: 17,084 train, 2,146 validation and 2,158 test under official patient folds. The 411 without mapped diagnostic targets remain available as images. PNG files total 3.37 GB. The new model uses its own multilabel runner and full-lead image preprocessing; the original four-class runner and whole-page resize are not used for PTB-XL.

**Completed on 23 September 2026:** 928 source images were audited, 437 redundant copies excluded, and 491 unique images split 295/98/98 with printed-ID grouping. LDA and CNN both ran successfully on CPU.

| Executed model | Test accuracy | Balanced accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|
| PCA + LDA | 62.24% | 60.32% | 0.5637 | 0.6423 |
| Small CNN | 75.51% | 56.86% | 0.5621 | 0.7310 |
| CNN–LSTM (spatial image features) | 71.43% | 63.15% | 0.6020 | 0.7168 |

The CNN's accuracy gain comes with poorer MI and History_MI recall; it is not a general improvement across classes. Only six MI examples occur in the test set. See the supervisor report for the scientific interpretation and limitations.

**Earlier Mendeley experiment, 24 September 2026:** [CNN–LSTM report](cnn_lstm_report.md), [three-model comparison](../results/comparisons/cnn_lstm_image_v1_seed42/comparison.md), and [PTB-XL migration assessment](ptbxl_migration_assessment.md). CNN–LSTM processes a spatial sequence of image features, not digitized ECG time samples. It improves macro F1 and balanced accuracy on this split, while overall accuracy falls compared with CNN. Its checkpoint-selection rule also differs; this does not isolate the effect of adding LSTM. At that completion checkpoint, all 21 tests passed and the central log contained three completed experiments. Those results remain unchanged.

**Earlier review on 24 September 2026:** development error analysis, label/group consistency checks, and an eight-image training input inspection are saved in [the pre-implementation handoff](next_model_handoff.md). The original CNN correctly classified only 1/6 MI and 0/17 History_MI validation images. One inspected crop cuts off waveform peaks; a separate one-case preview is not an adopted preprocessing fix. Clinical labels and the exact LSM definition remain unverified. That review snapshot preceded CNN–LSTM. Use `/opt/anaconda3/bin/python -m src.baseline_preservation` for current preservation checks; the older `src.check_readiness` protocol still concerns unresolved LSM and accepts append-only experiment-log additions.

## Results and supervisor materials

- `docs/supervisor_report.md`: findings, executed results, limitations, and recommended discussion.
- `results/comparisons/comparison.md` and `comparison.png`: same-split LDA/CNN comparison.
- `data/audit/dataset_audit.json` and `data/splits/fixed_split.csv`: exact data audit and frozen split.
- `experiments/experiment_log.csv`: completed and failed model runs. Future model runners must append here.
- `docs/existing_project_audit.md`: audit of recovered work and historical evidence.
- `docs/dataset_audit.md` and `docs/final_verification.md`: dataset findings and artifact checks.
- `docs/model_progression.md`: supervisor's requested sequence; LSM and the remembered convolution term remain unconfirmed.
- `docs/cnn_lstm_report.md`: completed third experiment, interpretation, verification and reproduction commands.
- `docs/ptbxl_migration_assessment.md`: local waveform/generated-image inventory and dataset-transition requirements.
- `docs/ptbxl_image_dataset.md`: completed PTB-XL rendering, labels, official patient partitions, verification and scientific limits.
- `docs/ptbxl_cnn_lstm_report.md`: completed five-label image experiment, per-label results, uncertainty and verification.
- `docs/teams_update_ptbxl_cnn_lstm.txt`: concise supervisor update ready to copy into Teams.
- [Spatiotemporal protocol](ptbxl_spatiotemporal_protocol.md) and [report](ptbxl_spatiotemporal_report.md): completed fifth experiment, fixed method and per-label evidence.
- [PTB-XL paired comparison](../results/comparisons/ptbxl_spatiotemporal_image_v1_seed42/comparison.md): saved-prediction differences, patient-bootstrap uncertainty and model limitations.
- [Latest supervisor update](teams_update_ptbxl_spatiotemporal.txt): completed comparison and proposed next research steps.
- `results/ptbxl_images_v1/dataset_overview.png`: classification partition and overlapping label counts; SVG is also saved.
- `experiments/dataset_log.csv`: dataset-generation record, kept separate from model runs.

## Environment

The executed environment uses Python 3.12 on macOS arm64, with the versions pinned in `requirements.txt`. Its interpreter is `/opt/anaconda3/bin/python`. The CNN probes MPS and falls back to CPU if unavailable or unsupported. It requires neither CUDA nor a paid service.

For a separate environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run commands from this project directory. On this machine:

```bash
cd "/Users/miriamistanboli/Downloads/ecg swiss"
/opt/anaconda3/bin/python -m src.verify
/opt/anaconda3/bin/python -m src.run --model lda --run-id lda_v1_seed42
/opt/anaconda3/bin/python -m src.run --model cnn --run-id cnn_v1_seed42
/opt/anaconda3/bin/python -m src.compare --lda-id lda_v1_seed42 --cnn-id cnn_v1_seed42
```

These experiment IDs are the original runs and cannot be overwritten. For a deliberate reproduction, use new IDs such as `lda_reproduction_seed42` and `cnn_reproduction_seed42`, then pass those IDs to `src.compare`. Do not adjust hyperparameters after examining test results. Reproducing an unchanged model is distinct from tuning against the test set.

The comparison command only reads saved predictions and checks their metrics; it does not train models or evaluate checkpoints again.

For the completed CNN–LSTM protocol, use a new run ID for deliberate reproduction:

```bash
/opt/anaconda3/bin/python -m src.run_cnn_lstm --run-id cnn_lstm_image_reproduction_seed42
/opt/anaconda3/bin/python -m src.compare_cnn_lstm --cnn-lstm-id cnn_lstm_image_reproduction_seed42
```

The original third run is `cnn_lstm_image_v1_seed42`. Its default configuration is `experiments/configs/cnn_lstm_image_v1.json`. Training is CPU-only, deterministic on the recorded runtime, and selects validation macro F1 with prespecified tie-breaks. Existing run IDs and comparison outputs cannot be overwritten. The central log is appended while its original two baseline rows are preserved exactly.

The completed PTB-XL CNN–LSTM run is `ptbxl_cnn_lstm_image_v1_seed42`; its configuration is `experiments/configs/ptbxl_cnn_lstm_image_v1.json`. Both PTB-XL models use the immutable input cache `data/processed/ptbxl_lead_images_v1/`. Their reports document preparation, execution and verification.

The fifth run used the following training invocation, recorded here for reproducibility. **It is historical: the existing run ID and frozen protocol must not be reused for a new training run.**

```bash
/opt/anaconda3/bin/python -m src.run_ptbxl_spatiotemporal --config experiments/configs/ptbxl_spatiotemporal_image_v1.json --run-id ptbxl_spatiotemporal_image_v1_seed42
```

The frozen spatiotemporal protocol binds an exact **four-row prior experiment-log snapshot**. The current log has five completed rows, so changing the run ID alone does not authorize or enable a fresh full run. A future training repetition needs a separately frozen protocol/configuration and preservation inventory with the updated prior-log baseline, plus a new run ID. Preserve the existing protocol and all five completed runs.

Saved-artifact verification and an offline comparison can be reproduced without model inference:

```bash
/opt/anaconda3/bin/python -B -c "from src.verify_ptbxl_spatiotemporal import verify; verify('ptbxl_spatiotemporal_image_v1_seed42'); print('Saved run verified without rewriting artifacts')"
/opt/anaconda3/bin/python -m src.compare_ptbxl_models --old-run-id ptbxl_cnn_lstm_image_v1_seed42 --new-run-id ptbxl_spatiotemporal_image_v1_seed42 --output-dir results/comparisons/ptbxl_spatiotemporal_recheck
```

The comparison requires a new empty output directory and reads only saved predictions. Subsequent same-cohort model comparisons remain exploratory because the test results are already observed; future choices and any repeated-seed study must be specified using development evidence.

## Dataset preparation

The separate PTB-XL image cohort is in `data/derived/ptbxl_images_v1/`. The original one-time generation used `requirements-ptbxl-images.txt` and `src.generate_ptbxl_images`; the completed dataset already exists. For current integrity checks use `src.audit_publication_data` above. Do not rerun the historical generator verification command on this frozen dataset: it rewrites the bound `verification.json` timestamp. Exact generation source/configuration snapshots are preserved under `experiments/protocols/ptbxl_images_v1/source_snapshot/`. Read its dataset card before future model preprocessing; the Mendeley preparation below does not apply to it.

The original image archive has 928 source JPEGs. An additional hidden notebook checkpoint is not a source sample. Exact duplicates are excluded by decoded-pixel hash, with a deterministic representative; originals are not deleted. Every exclusion and its retained representative is recorded in `data/audit/exclusions.csv`.

Printed header IDs are extracted locally and used as conservative grouping constraints. They are **not verified patient identifiers**. A printed ID shared by two pages cannot cross partitions. Exact cropped-input collisions and audited near-copy links also cannot cross partitions. Unresolved printed IDs are excluded with a recorded reason.

One prespecified, shuffled, stratified group split (seed 42; first of five outer folds, first of four inner folds) targets 60% training, 20% validation, and 20% test. Group constraints can prevent exact proportions and class ratios. The split is never searched for higher performance.

On a fresh copy of the original data, the one-time preparation commands are:

```bash
python -m src.audit_archive
python -m src.audit_near_duplicates
# Uses the saved manual_id_validation.csv and visually annotated reference pages.
python -m src.preprocessing.template_header_ids
python -m src.prepare --config experiments/configs/baseline_v1.json
python -m src.verify
```

`src.prepare` refuses to replace an existing split. The saved split and its audit files are the authority for these runs; rebuilding a different audit is a new dataset/split version. Distribute the original source data and necessary local audit artifacts through an appropriate private channel when reproducing, since raw images, OCR identity mappings, caches, and checkpoints are intentionally excluded from Git. Do not infer unavailable original patient identities from the filenames.

The executed header method is an offline exact-glyph reader using NumPy, Pillow, and SciPy. All 2,946 digits on 491 pages exactly match templates, and 120 visual checks covering 115 distinct source hashes agree. This is a narrow reader for this dataset's fixed typography, not a general OCR tool. The unused Vision helper and optional macOS requirements are preserved as documented failed audit attempts; they are not required to reproduce the baselines.

## Methods

The three Mendeley models receive the same fixed crop `[0.03, 0.19, 0.98, 0.96]` in normalized left/top/right/bottom coordinates, then grayscale bilinear resizing to 128×256 and inversion to ink intensity in [0,1]. This removes the header and footer; grid and lead-label artifacts remain. There is no SNR filtering, augmentation, or unvalidated waveform calibration. The crop is a small computational image baseline, not a clinically lossless representation.

- **LDA:** flatten 32,768 pixel values; fit centered randomized PCA with 32 components on training samples only; fit `LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")` with uniform class priors. No class-dependent test processing or validation-driven tuning.
- **CNN:** 40,212 parameters; three convolution stages (16/32/64 channels), GroupNorm, coarse 2×4 adaptive pooling, dropout, and a small classifier. Global mean/std and inverse-frequency class weights come only from training. AdamW; at most 30 epochs; early stopping after seven epochs without validation-loss improvement; restore the minimum-validation-loss checkpoint.
- **CNN–LSTM:** 44,532 parameters; the same three convolution blocks, pooling to two vertical bands and 16 horizontal positions, a one-layer LSTM with 32 hidden units, mean sequence output and four-class classifier. Uses the same frozen images and train-only statistics; no waveform digitization. Checkpoint selection maximizes validation macro F1, then balanced accuracy, then minimizes weighted loss. One configuration/seed, 30 epochs maximum, patience seven and gradient clipping at 1.0. See the report for comparison limits.

The three Mendeley models save accuracy, balanced accuracy, per-class precision/recall/F1/support, macro and weighted F1, confusion matrix, valid multiclass one-vs-rest ROC-AUC, and per-sample probabilities. Their neural models additionally save epoch histories, learning curves, architecture, and normalization in their checkpoints. All runs save configuration, environment, source hashes, split/cache hashes, timing, and failure evidence where applicable.

The **PTB-XL image CNN–LSTM** is a separate 18,025-parameter model. It retains all twelve complete row bands, including calibration pulses, inverts grayscale to ink, and applies isotropic 2×2 maximum pooling to form twelve 120×1120 image channels. Global mean/std come only from training pixels; compressed cached PNGs are loaded lazily. A small CNN feeds 70 left-to-right image-feature positions to a one-layer LSTM; five independent sigmoid probabilities support overlapping labels. It uses unweighted binary cross-entropy, one seed/configuration, at most 20 epochs, and patience four. Checkpoint selection maximizes validation macro AUROC, then macro average precision, then minimizes probability BCE; the decision threshold remains 0.5. The sequence includes calibration/margin pixels and is not recovered 500 Hz waveform data. Maximum pooling preserves dark strokes but thickens them and merges fine detail. No native waveform values or numeric provenance features enter the model.

The **PTB-XL image spatiotemporal model** has 17,161 parameters and keeps that input representation and CNN encoder. It projects each of the 70 image-feature vectors from 96 to 32 channels, applies two residual temporal-convolution blocks with dilations 1 and 2, then averages all positions for five independent label outputs. Its head has a 13-feature-position receptive field. Left padding is causal only within the head's feature sequence; the spatial CNN and global averaging do not provide causal waveform or streaming inference. Training retains seed 42, CPU execution, the same optimizer/epoch budget, validation selection and fixed threshold. Different head capacity and dropout placement prevent a pure mechanism-only claim about convolution versus LSTM.

## Validation and project structure

```bash
python -m unittest discover -s tests -v
python -m pytest -q tests/test_publication_data_audit.py
```

The focused tests check known metric values, leakage protection, duplicate grouping, immutable artifacts, train-only preprocessing, deterministic CNN behavior, and checkpoint reconstruction.

```text
data/raw/          references to preserved original data
data/audit/        source manifest, exclusions, identity/duplicate audit
data/processed/    compact shared input cache (ignored by Git)
data/splits/       frozen sample/group assignments and protocol
data/derived/      separately versioned PTB-XL pages, labels, calibration and official splits
src/               reusable preparation, model, evaluation, comparison code
experiments/       fixed config, central CSV log, execution logs
models/lda/        fitted PCA/LDA pipeline
models/cnn/        checkpoint, architecture, history, training curve
models/cnn_lstm/   spatial-sequence model checkpoint and training evidence
models/ptbxl_cnn_lstm/  five-label lead-image model checkpoint and training evidence
models/ptbxl_spatiotemporal/  five-label image CNN/temporal-convolution checkpoint and training evidence
results/           predictions, metrics, confusion matrices, comparison
docs/              project audit, supervisor report, progression
notebooks/         reserved; not the source of experiment logic
```

No original work was deleted or moved. The historical `.ipynb_checkpoints/weirdwithseed-checkpoint.py` is preserved for audit and should not be used as the new baseline runner.
