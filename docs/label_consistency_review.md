# Label consistency and readiness review

The technical label review is complete. The supplied folder labels remain unchanged; clinical label validity is unresolved. This review verifies provenance, duplicate-label agreement, grouping and queue coverage. It does not assign diagnoses or decide that a record is mislabeled.

Frozen dataset: `image_dedup_8e04bd56af2f`. Frozen split: `grouped_cf5a867a99c3`. Reproduce with `python3 -m src.review_labels` from the project root; only standard-library Python is required.

## Verified evidence

All 928 enumerated raw image hashes were rechecked against the source manifest. The retained 491 canonical paths, labels, sample identifiers and hashes agree with the fixed split. All 437 exclusions map to a retained representative with the same pixel hash and folder label. The fixed split, configuration, input cache, metadata and grouping-evidence hashes agree with the frozen audit and remained unchanged during this review.

There are 0 exact-file duplicate groups, 0 decoded-pixel duplicate groups and 0 identical-model-input groups containing conflicting folder labels. The archive audit already records 928 byte-for-byte matches between the ZIP and source files; this review rechecks the current files against their frozen manifest rather than treating the archive as an independent cohort.

The existing near-copy screen considered 1,508 pairs and accepted 0 extra grouping links. Its fixed-crop perceptual-hash, correlation and absolute-difference thresholds may miss shifted, rescaled or redrawn copies. No absence-of-near-copies or patient-independence claim follows from a zero-link result.

## Distinct identifier groups

There are 311 printed-ID proxies and 311 frozen groups; no group combines multiple distinct printed IDs in this cohort. Equal printed IDs remain together in one partition, and no frozen group crosses partitions. These are separate identifier groups, not confirmed independent patients.

| Partition | Records | Distinct groups | Groups spanning folder labels | Records in those groups | Groups with multiple records |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 295 | 187 | 38 | 133 | 49 |
| validation | 98 | 63 | 12 | 41 | 18 |
| test | 98 | 61 | 12 | 46 | 15 |
| all | 491 | 311 | 62 | 220 | 82 |

Per-class counts of distinct groups overlap: a group represented in two labels contributes once to each of their rows. Summing a partition's class-level group counts therefore overstates its total distinct groups.

| Partition | Supplied label | Records | Distinct ID / frozen groups | Single-label groups | Groups also containing another label |
| --- | --- | ---: | ---: | ---: | ---: |
| train | Normal | 85 | 70 | 43 | 27 |
| train | MI | 18 | 18 | 8 | 10 |
| train | Abnormal | 140 | 104 | 73 | 31 |
| train | History_MI | 52 | 45 | 25 | 20 |
| validation | Normal | 29 | 25 | 15 | 10 |
| validation | MI | 6 | 5 | 4 | 1 |
| validation | Abnormal | 46 | 33 | 23 | 10 |
| validation | History_MI | 17 | 14 | 9 | 5 |
| test | Normal | 28 | 22 | 13 | 9 |
| test | MI | 6 | 6 | 3 | 3 |
| test | Abnormal | 47 | 36 | 25 | 11 |
| test | History_MI | 17 | 14 | 8 | 6 |
| all | Normal | 142 | 117 | 71 | 46 |
| all | MI | 30 | 29 | 15 | 14 |
| all | Abnormal | 233 | 173 | 121 | 52 |
| all | History_MI | 86 | 73 | 42 | 31 |

The overall group compositions are:

| Supplied labels within group | Groups | Records |
| --- | ---: | ---: |
| Abnormal | 121 | 133 |
| Abnormal+History_MI | 10 | 31 |
| History_MI | 42 | 45 |
| MI | 15 | 16 |
| MI+Abnormal | 5 | 10 |
| MI+History_MI | 1 | 2 |
| Normal | 71 | 77 |
| Normal+Abnormal | 22 | 66 |
| Normal+Abnormal+History_MI | 9 | 58 |
| Normal+History_MI | 7 | 14 |
| Normal+MI+Abnormal | 4 | 19 |
| Normal+MI+Abnormal+History_MI | 2 | 13 |
| Normal+MI+History_MI | 2 | 7 |

A repeated ID spanning folder labels is a provenance question, not proof of a wrong label. Possible explanations include different examinations, changes over time, identifier reuse, source-label conventions, or inconsistent labeling. The available metadata does not distinguish these explanations. No automatic majority vote, relabeling or group deletion was performed.

## Development-only adjudication queue

`results/readiness/labels/development_adjudication_queue.csv` contains every one of the 393 canonical training/validation records from 250 groups. It contains zero test records. Each row preserves its canonical source path, sample ID, opaque group ID, supplied label and file/pixel/model-input hashes. It includes no model prediction, confidence, error flag or ranking based on model results.

Rows from mixed-folder groups are listed first, followed by the small MI class and routine provenance checks. This ordering directs review effort without claiming an error. The fields `verified_label`, `reviewer`, `evidence` and `status` are all blank: a prepared queue is not a completed adjudication. The script refuses to overwrite the queue once any of these fields has been filled.

A qualified reviewer should record the available original evidence, verify what the printed ID represents, and assess the supplied categories using a documented labeling policy. Review should allow an unresolved outcome when the source information is insufficient. The four broad folder categories do not establish a unified clinical taxonomy; an ECG page alone may not support the clinical history encoded by a folder name. Metadata consistency and accurate header transcription cannot clinically validate these labels.

Any accepted corrections require a separately authorized, versioned label revision with its evidence and a declared evaluation policy. Preserve this frozen baseline cohort for comparison; do not silently edit its labels or tune corrections using test errors. The current review exports test information only as group-level technical metadata and aggregate counts, and creates no test adjudication queue.

## Readiness and artifacts

The dataset is technically ready for a prespecified next exploratory model on the same frozen split, subject to the project's model-definition decision. Clinical-ground-truth validation remains open. Results must continue to be described as prediction of supplied dataset categories, with limited minority-class support, uncertain patient identity and source-label limitations.

- `results/readiness/labels/group_review.csv`: all opaque groups, partition, record counts and supplied-label composition; no printed ID values or model outcomes.
- `results/readiness/labels/class_group_counts.csv`: distinct ID/group counts per class and partition, including mixed-group overlap.
- `results/readiness/labels/group_compositions.csv`: group-composition counts for every partition and the complete cohort.
- `results/readiness/labels/development_adjudication_queue.csv`: unfilled training/validation review form with original paths and hashes.
- `results/readiness/labels/summary.json`: computed findings, limitations and input hashes.
- `src/review_labels.py`: reproducible review, integrity checks and queue generation.

Local evidence: `docs/dataset_audit.md`, `docs/source_references.md`, `data/audit/archive_provenance.json`, `data/audit/header_template_summary.json` and `data/audit/near_duplicate_summary.json`. The documented source attribution does not substitute for patient-level source metadata or independent label adjudication. No model outputs were opened by the helper and no external service was contacted.
