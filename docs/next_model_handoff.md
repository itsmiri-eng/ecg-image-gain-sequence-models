# Baseline review and next-model handoff — 24 September 2026

**Historical review snapshot:** the user subsequently chose CNN–LSTM, which is now completed under a separate spatial-image protocol. See [CNN–LSTM report](cnn_lstm_report.md) for current results. LSM remains unconfirmed. The text below records what was known before that implementation; its “next model” status is not the status of the now-completed CNN–LSTM.

The technical follow-up to the LDA and CNN baselines is complete: saved development errors were reviewed, label/group consistency was checked, input quality was spot-checked, and a guarded next-experiment protocol was prepared. **The next model has not been implemented or run.** Its exact name and definition remain unresolved. Clinical label verification and cohort-wide validation of replacement preprocessing are also unfinished.

## What the review established

The saved validation predictions show that the CNN's higher overall accuracy does not extend to the minority classes:

| Validation measure | PCA + LDA | CNN |
|---|---:|---:|
| Accuracy | 62.24% | 71.43% |
| Balanced accuracy | 54.48% | 50.91% |
| Macro F1 | 0.5296 | 0.4607 |
| MI correctly classified | 2 / 6 | 1 / 6 |
| History_MI correctly classified | 9 / 17 | 0 / 17 |

The CNN assigns 92 of 98 validation images to Normal or Abnormal. At its saved checkpoint it also correctly classifies only 1 of 18 training MI images and none of 52 training History_MI images. This is uneven class learning, and cannot be explained simply as good training performance followed by overfitting. The review does not establish whether labels, representation, optimization or class overlap caused it. See [minority error review](minority_error_review.md) and the saved [validation confusion figure](../results/readiness/errors/validation_confusion.png).

All 928 original image hashes were reverified. No conflicting supplied labels were found among identical files, decoded images or model inputs. However, 62 of 311 printed-ID groups contain more than one folder label, covering 220 images. These IDs are unverified report/patient proxies; mixed labels are review flags, not proof of incorrect diagnoses. The six validation MI images represent only five such groups.

A [label-review queue](../results/readiness/labels/development_adjudication_queue.csv) contains all 393 training/validation images, with no test records or model predictions. Clinical reviewer, evidence and verified-label fields are blank. A qualified reviewer or authoritative source is still needed to adjudicate them; no labels were changed. See [label consistency review](label_consistency_review.md).

## Input issue and bounded preview

Eight deterministically selected training images were inspected. Their preprocessing exactly reproduces the frozen cache. One, `Abnormal/HB(59).jpg`, has four tall trace peaks extending above the fixed crop: 66–76 source pixels of their vertical extent are omitted. The resize also stretches horizontal geometry by 15.22% relative to vertical geometry and reduces fine detail. This small inspection does not estimate defect prevalence or certify the other images.

A separate [candidate preview](../results/readiness/inputs/candidate_hb59_comparison.png) demonstrates a higher crop boundary and aspect-preserving resize with padding for this one case. It retains the identified peaks, but may retain technician/referring-physician text. It has **not** replaced the cache, been validated across the cohort, or been used to train a model. Its crop, resolution and resize geometry all differ from baseline. A future preprocessing experiment must validate trace preservation and header exclusion on development data, version its inputs, and compare both baselines under the same replacement representation. See [input representation review](input_representation_review.md).

## What is preserved and verified

The 491-image dataset, seed-42 split (295 training / 98 validation / 98 test), labels, fitted models, saved results and original model code remain unchanged. The preservation check passes for all 67 locked baseline artifacts. The central experiment log still contains exactly the two completed model runs. Their original test accuracies remain 62.24% for LDA and 75.51% for CNN; original test macro F1 remains 0.5637 and 0.5621 respectively.

This follow-up uses saved development predictions and image previews. It performs no model fitting or new inference. Previously published test aggregate metrics are retained only as context; there is no new test-case gallery or test-guided tuning. The already observed test set cannot provide a fresh confirmatory assessment of subsequent research choices.

## Next experiment

1. Obtain the exact full name and sufficient definition/reference for **LSM**, the next unresolved item in the supervisor's progression. The remembered convolution term also remains unresolved. Neither is inferred from its sound or abbreviation.
2. Freeze a method-specific configuration, small local compute budget, validation selection rule and deterministic tie-break before fitting. The prepared [protocol](../experiments/configs/next_model_protocol.json) proposes validation macro F1 as the primary development measure, with balanced accuracy and per-class recall reported alongside it. This does not retroactively change baseline checkpoint selection.
3. Use the existing split for an explicitly limited comparison. Any changed preprocessing is a separate experimental factor and requires its own development validation and version; the one-case crop preview is not ready for adoption. Fit all learned preprocessing on training data only.
4. After freezing the candidate, save its checkpoint, configuration, runtime, development metrics, final exploratory test predictions and central experiment-log entry. Do not repeatedly tune against test results. New confirmatory claims require independent untouched data or a properly nested group-aware evaluation.

CNN–LSTM remains planned. Pixel columns are not validated time samples. A spatial sequence of image features must be described as such; a temporal waveform model requires validated lead extraction and calibration first.

## Recheck the handoff

From the project directory:

```bash
/opt/anaconda3/bin/python -m src.check_readiness
```

This verifies baseline preservation, frozen data integrity and both development queues, and writes `results/readiness/readiness.json`. Adding `--require-model` currently exits with code 2 because the exact method and its training specification are pending; this is expected, not a failed model experiment. The read-only check does not run a model.

New review artifacts and their hashes are listed in `experiments/reviews/2026-09-24/review_log.json`. Original baseline verification and delivery records remain historical snapshots.
