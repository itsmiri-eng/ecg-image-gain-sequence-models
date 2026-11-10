# Minority-class development error review

This review uses the already saved LDA/CNN predictions. It performs **no inference, retraining, calibration, threshold adjustment, or model selection**. Case-level analysis and images are limited to training and validation. Folder labels are described as supplied labels, not independently adjudicated diagnoses.

## Validation findings

| Supplied class | Images | LDA correct | CNN correct | CNN error destinations |
|---|---:|---:|---:|---|
| MI | 6 | 2 | 1 | Normal: 3, Abnormal: 2 |
| History_MI | 17 | 9 | 0 | Normal: 12, MI: 1, Abnormal: 4 |

CNN predicts Normal or Abnormal on **92/98 validation images (93.9%)**, compared with **68/98 (69.4%)** for LDA. The supplied validation labels contain 75/98 Normal or Abnormal images. CNN's validation accuracy is higher (71.4% vs 62.2%), while balanced accuracy is lower (50.9% vs 54.5%) and macro F1 is lower (0.461 vs 0.530). These conclusions use validation evidence alone.

The same minority failure occurs in training: CNN correctly predicts **1/18 MI** and **0/52 History_MI** images, compared with LDA **13/18** and **45/52**. Both models nevertheless have **213/295 training predictions correct**. CNN predicts Normal or Abnormal on **276/295 training images (93.6%)**. This is evidence of uneven class learning; it does not isolate whether representation, optimization, sample size, label meaning, or image artifacts cause it. Minority failure is not explained by a train-to-validation accuracy gap alone.

## Supplied-label rank and shared errors

Ranks below describe the existing probability order; they do not change the classifier or recommend a new decision rule.

| Validation class | LDA supplied label in top 2 | CNN supplied label in top 2 |
|---|---:|---:|
| MI | 4/6 | 3/6 |
| History_MI | 15/17 | 13/17 |

Across all validation images, both models are wrong on **16/98**, LDA alone on **21/98**, CNN alone on **12/98**, and both correct on **49/98**. They select the same wrong label on **10 of their 16 shared errors**. For History_MI, 8 are shared errors and 9 are CNN-only errors; neither model pair is jointly correct on a History_MI case. For MI, 3 are shared errors, 1 is LDA-only, and 2 are CNN-only, with no jointly correct case. Agreement is not label validation.

## Group label diversity: descriptive association

A flagged group contains multiple supplied folder labels. Equal report IDs were already grouped before splitting; their relationship to patients remains unverified. Different labels within a group can reflect different recordings or inconsistent labels, so the flag does not establish an error.

| Validation scope | Multi-label group? | Images / groups | LDA errors | CNN errors |
|---|---|---:|---:|---:|
| all classes | No | 57 / 51 | 25 | 19 |
| all classes | Yes | 41 / 12 | 12 | 9 |
| minority combined | No | 15 / 13 | 9 | 14 |
| minority combined | Yes | 8 / 6 | 3 | 8 |
| MI | No | 5 / 4 | 4 | 4 |
| MI | Yes | 1 / 1 | 0 | 1 |
| History MI | No | 10 / 9 | 5 | 10 |
| History MI | Yes | 7 / 5 | 3 | 7 |

These are descriptive counts, not a causal or significance analysis. Images within groups are dependent, group sizes and class composition differ, and some cells are very small. MI has only six validation images. No diagnosis is inferred from the group association.

## Already exposed test results: post-hoc aggregate context only

The test prediction CSVs were not read for this review. Existing test confusion matrices reproduce the recorded aggregates: LDA correctly predicts MI **3/6** and History_MI **10/17**; CNN **1/6** and **4/17**. CNN predicts Normal or Abnormal on **88/98** test images. This context does not drive changes or examples here. The test set has already been exposed; further models on this cohort remain exploratory, not fresh unbiased validation.

## Review artifacts and limits

- `results/readiness/errors/development_cases.csv`: 393 training/validation cases, supplied labels, paths, hashes, groups, saved predictions/probabilities, supplied-label ranks, and review flags. It contains no test cases and is unblinded; keep it separate from the independent blind label queue.
- `partition_class_metrics.csv`, `confusion_counts.csv`, and `minority_error_destinations.csv`: reproducible partition aggregates with an explicit post-hoc role on test rows.
- `development_error_overlap.csv`, `development_supplied_label_ranks.csv`, and `development_group_associations.csv`: development-only tables.
- `validation_confusion.png` and `development_prediction_counts.png`: development figures.
- `validation_minority_inputs.png` and `validation_gallery_index.csv`: deterministic first six sample IDs per minority class, at most 12 validation crops identical to model inputs. Headers are excluded by the frozen crop. Inspect only clipping, layout, grid/text remnants, or other input-quality features; no diagnoses or new labels were assigned.
- `summary.json`: machine-readable counts, source hashes, scope flags, and verification that all 67 locked baseline artifacts are unchanged.

Before future development, the findings justify documenting class-wise failure and independently reviewing label meaning and input quality. They do not justify a particular architecture, calibration procedure, threshold, or claimed remedy. Predeclare any next experiment using development evidence and retain the existing baselines unchanged.

Reproduce after explicitly archiving this review's new outputs:

```bash
/opt/anaconda3/bin/python -m src.review_errors
```
