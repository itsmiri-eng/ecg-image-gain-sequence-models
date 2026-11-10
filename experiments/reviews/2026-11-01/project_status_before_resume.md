# Project status — 26 September 2026

**Fifth experiment completed and verified:** `ptbxl_spatiotemporal_image_v1_seed42` completed at **23:17 UTC on 25 September 2026**, which is 26 September locally. It ran 20 epochs and restored validation-selected **epoch 16**. The 17,161-parameter CNN with a temporal-convolution head keeps the previous PTB-XL inputs, patient folds, five labels and training/selection budget. Test macro AUROC improved from **0.8698 to 0.8803**, paired difference **+0.0105** (95% patient-bootstrap interval **+0.0045 to +0.0167**); macro F1 improved from **0.6131 to 0.6409** at threshold 0.5. See the [frozen protocol](</Users/miriamistanboli/Downloads/ecg swiss/docs/ptbxl_spatiotemporal_protocol.md>), [completed report](</Users/miriamistanboli/Downloads/ecg swiss/docs/ptbxl_spatiotemporal_report.md>) and [paired comparison](</Users/miriamistanboli/Downloads/ecg swiss/results/comparisons/ptbxl_spatiotemporal_image_v1_seed42/comparison.md>).

The spatiotemporal model is the **provisional stronger overall PTB-XL candidate**, with validation macro AUROC also improving from 0.8816 to 0.8889. It is not uniformly better: test MI recall rose from **43.27% to 56.18%**, but HYP recall fell from **32.44% to 27.86%**, leaving **189 of 262 HYP positives missed**. HYP recall also fell on validation. The test cohort was already observed before this experiment, so this is an exploratory comparison, not a fresh confirmatory evaluation. Patient-bootstrap uncertainty excludes training-seed and external-domain variation.

The fifth run used one fixed configuration/seed and records **one final test inference** after selection. Saved-artifact verification passed and all **152 tests pass**. The **164 protected prior files and four prior central-log rows are preserved**; the central log now contains five completed experiments. Inputs remain twelve full lead rows with calibration, 2×2 maximum ink pooling and training-only normalization. Downsampling loss, variable gain and untested real-scan generalization remain material limits. Recorded elapsed timers are not pure CPU time: Mac sleep affected this execution, whose UTC wall span is reported separately. No controlled speed advantage is established.

**Earlier completed PTB-XL model — 24 September 2026:** `ptbxl_cnn_lstm_image_v1_seed42` stopped early after 19 epochs and restored validation-selected epoch 15. Test macro AUROC is **0.8698** (95% patient-bootstrap interval **0.8593–0.8805**), macro average precision **0.7035**, and macro F1 **0.6131** at threshold 0.5. Exact five-label match accuracy is **52.59%** and individual label-decision accuracy **84.95%**. At that completion checkpoint, all 112 tests passed and the central log contained four completed runs. These results remain unchanged; see the [PTB-XL CNN–LSTM report](ptbxl_cnn_lstm_report.md) and [saved verification](../results/ptbxl_cnn_lstm/ptbxl_cnn_lstm_image_v1_seed42/verification.json). Neither PTB-XL model's scores are directly comparable to the earlier four-class Mendeley results.

**Dataset status:** All 21,799 PTB-XL pages and provenance records are saved and verified; 21,388 have mapped multilabel targets (17,084 train / 2,146 validation / 2,158 test). Official patient folds are retained. The 411 unmapped records remain as images, excluded only from classification. Every source PNG passed decoding and hash/source checks; no exact rendered-image duplicates or patients crossing partitions were found. PNG files occupy 3.37 GB. See the [dataset card](ptbxl_image_dataset.md), [overview figure](../results/ptbxl_images_v1/dataset_overview.png), and [dataset verification](../data/derived/ptbxl_images_v1/verification.json).

**Next research step:** test whether the promising overall result is stable across prespecified repeated seeds using development data, then assess a frozen approach on independent data. Neither study has been performed. Any changes to thresholds, imbalance handling or preprocessing need a separate development protocol; HYP weakness is not permission to tune on the saved test predictions. A fresh full run also needs a new protocol/preservation baseline: the completed spatiotemporal protocol locks the prior four-row log and cannot simply accept a new run ID now that the log has five rows. LSM and the remembered convolution term still require exact definitions; ViT remains deferred. See the [research direction](</Users/miriamistanboli/Downloads/ecg swiss/docs/research_direction.md>) and [latest supervisor update](</Users/miriamistanboli/Downloads/ecg swiss/docs/teams_update_ptbxl_spatiotemporal.txt>).

## Historical Mendeley CNN–LSTM completion

The earlier image experiment achieved test accuracy 71.43%, balanced accuracy 63.15%, macro F1 0.6020 and weighted F1 0.7168. At that checkpoint, all 21 tests passed, checkpoint training replay was exact, and the central log contained three completed runs. See [CNN–LSTM report](cnn_lstm_report.md) and [PTB-XL assessment](ptbxl_migration_assessment.md). The sections below preserve earlier baseline and review snapshots, rather than the current overall status.

## Historical two-model baseline delivery — 23 September 2026

The immediate deadline deliverable is complete: two actual model runs, an audited frozen dataset/split, saved evidence, and a supervisor report.

| Work item | Status | Evidence |
|---|---|---|
| Existing-project audit | Complete for local files; referenced remote GitHub contents unverified | `existing_project_audit.md`, `source_references.md` |
| Image integrity, exact duplicates, archive provenance | Complete | `dataset_audit.md`, `../data/audit/` |
| Printed-ID audit | Complete for all 491 unique pages; patient semantics unverified | `../data/audit/header_template_summary.json` |
| Fixed train/validation/test split | Complete, seed 42; 295/98/98 images | `../data/splits/fixed_split.csv`, `protocol.json` |
| LDA execution | Complete | `../results/lda/lda_v1_seed42/run.json` |
| CNN execution | Complete; 30 epochs, selected epoch 25 | `../results/cnn/cnn_v1_seed42/run.json` |
| Comparison | Complete; matched saved test records and metrics | `../results/comparisons/` |
| Experiment log and reproducibility | Complete | `../experiments/experiment_log.csv`, `../README.md`, `../requirements.txt` |
| Validation | Six focused tests pass; split and saved result checks pass | `../experiments/logs/` |
| LSM / uncertain convolution method | Awaiting exact terminology; deliberately not guessed | `model_progression.md` |
| CNN+LSTM / spatiotemporal / ViT | Not implemented or run | `model_progression.md` |

All original work remains in place. Large raw data were referenced rather than copied; the shared processed cache is about 11 MB and the fitted model directory about 4.4 MB. No datasets or checkpoints were committed, and this workspace was not initialized as a new Git repository.

The new baselines replace unvalidated page-to-signal reconstruction with a documented direct-image experiment. Prior manuscripts are retained as historical claims, not relabeled as new evidence. The important open scientific issues are the small number of unique MI examples, unverified printed-ID semantics, overlapping folder-label identities, remaining image-layout artifacts, and the lack of external validation.

Operational failures were recorded: Swift required an unaccepted Xcode license; sandboxed Vision failed; a permission probe and optional remote repository retrieval were interrupted. Offline exact-glyph transcription resolved the identity audit using existing libraries. Both actual model runs succeeded; no failed model run is hidden.

## Historical follow-up — before CNN–LSTM implementation

Technical review is complete; see [next-model handoff](next_model_handoff.md). Added saved development error analysis and figures, label/group consistency tables and a 393-record development adjudication queue, an eight-training-image representation inspection, and a separate one-case crop preview. The inspection found confirmed waveform clipping in one case; the preview has not replaced the baseline inputs and needs broader development validation and header exclusion checks before adoption. No clinical labels were verified or changed.

The readiness checker verifies the frozen dataset/split, both development queues and preservation of 67 original baseline artifacts. No new model or inference was run. The next experiment is pending the exact LSM definition/reference and a method-specific training configuration; the unresolved convolution term remains a placeholder. `experiments/configs/next_model_protocol.json` records these open items and a proposed validation-macro-F1 selection policy for future experiments. Existing test results are already observed and subsequent comparisons must be described as exploratory.
