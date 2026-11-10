# Project status — 1 November 2026 (SUPERSEDED copy, preserved verbatim)

Superseded on 1 November 2026 evening by `docs/PROJECT_STATUS.md` after the hybrid ViT run completed. Preserved unchanged below.

## Reconstructed stopping point

Five formal experiments completed successfully. The last was `ptbxl_spatiotemporal_image_v1_seed42`, completed on 25 September 2026 at 23:17 UTC (26 September in Muscat): 20 epochs, validation-selected epoch 16. Its checkpoint, all partition predictions, metrics, training history, verification, comparison, seven reviewed figures and supervisor paragraph exist. The final delivery manifest was never written; this was a packaging gap, not an unfinished training run.

Fresh read-only verification on 1 November passed for both PTB-XL runs. The three Mendeley runs’ metrics recompute from saved predictions, and source/checkpoint preservation checks pass. All 152 existing tests pass. See [results audit](../experiments/reviews/2026-11-01/results_audit.json) and [repository inventory](../experiments/reviews/2026-11-01/repository_inventory.json). The full dataset/source audit also passed: all 21,799 source pages, 500 Hz signal/header pairs and sidecars, all 21,388 classifier cache images, source/split reconstruction and training-only normalization. See [dataset audit](../experiments/reviews/2026-11-01/dataset_audit_resume.md).

## Valid evidence and comparison boundaries

| Cohort/task | Completed model | Test accuracy definition/value | Macro F1 | Macro AUROC |
|---|---|---:|---:|---:|
| Mendeley, four mutually exclusive folder labels | PCA + LDA | Single-class accuracy 62.24% | 0.5637 | See saved metrics |
| Same Mendeley split | Small CNN | Single-class accuracy 75.51% | 0.5621 | See saved metrics |
| Same Mendeley split | CNN–LSTM | Single-class accuracy 71.43% | 0.6020 | See saved metrics |
| PTB-XL, five overlapping diagnostic targets | Image CNN–LSTM | Exact five-label match 52.59% | 0.6131 | 0.8698 |
| Same PTB-XL split | Image CNN + temporal convolutions | Exact five-label match 56.16% | 0.6409 | 0.8803 |

The three Mendeley models share 295/98/98 images drawn from 491 deduplicated pages; printed report IDs are grouping proxies, not verified patient identities. The two PTB-XL models share 17,084/2,146/2,158 images and official patient folds 1–8/9/10. No across-cohort accuracy ranking is valid. Normalization/PCA are fitted on training data only. Exact duplicates and known group overlaps were addressed; unverifiable original Mendeley patient identities remain a limitation.

The spatiotemporal model has the stronger aggregate PTB-XL result in this single-seed exploratory comparison: paired macro-AUROC gain +0.01046, 95% patient-bootstrap interval [+0.00447,+0.01674]. MI recall rises from 43.27% to 56.18%, but HYP recall drops from 32.44% to 27.86%. This does not establish a universal best model. The shared test cohort has already been observed, and further comparisons cannot be described as fresh confirmatory evidence.

## Problems and unfinished publication work

- Archived manuscript versions contain results not tied to these formal runs and some conflicting figure/sample-count claims. They are historical material, not the authority for the new paper.
- Both PTB-XL report manifests have one stale input hash for the later-edited dataset card; their saved output hashes and all model evidence still pass. Fresh audit evidence records this documentation-only change without overwriting historical reports.
- There is no Git repository or commit history configured. No Git state is invented; file/config/source hashes provide the current provenance.
- The old full-training runners deliberately bind earlier log snapshots and cannot all be reused merely by changing a run ID. The next protocol must preserve prior runs while supporting explicit new-ID reproductions.
- The legacy log lacks some publication fields; an expanded derived master log and cohort-specific tables are being prepared without changing its schema. [Paper experiment notes](../paper/EXPERIMENT_NOTES.md), a [month plan](../paper/PUBLICATION_PLAN.md) and corrected references now exist.
- Only one seed per architecture has run. Matched PTB-XL LDA/CNN baselines, robustness/ablations, independent image validation and final manuscript work remain outstanding.

## Next model decision

LSM and the remembered convolution method remain unconfirmed in the original notes; literature mentions do not establish what the supervisor intended. CNN–LSTM and the small spatiotemporal network are already complete. A compact hybrid image transformer is the next candidate to assess: retain the same CNN image-feature representation and test global self-attention under the same data/split/training budget. Its 23,881-parameter design passed synthetic compute and training checks before fitting; it is not a standard large pretrained ViT or an assumption that transformers will win.

Current hardware probe: Python 3.12.4 / PyTorch 2.9.1 on arm64 macOS; MPS and CUDA are unavailable to this runtime. CPU is the working backend. Run `ptbxl_hybrid_vit_image_v1_seed42` started at 11:28:50 UTC on 1 November (15:28:50 Muscat). It is actively fitting the training partition, with maximum 20 epochs and patience four. Configuration, nine source files and audit/test evidence were frozen before training. Pretraining verification passed 172 unittest cases and seven additional audit pytest cases. No new test result is available yet.
