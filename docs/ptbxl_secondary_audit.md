# Historical PTB-XL inventory — before image generation and training

The local PTB-XL 1.0.3 directory was audited independently before the new rendered-image cohort and model existed. At this audit stage, its waveform records were not mixed with the Mendeley image dataset, and no PTB-XL model had been trained. Subsequent work produced the [separate image cohort](ptbxl_image_dataset.md) and [completed PTB-XL image CNN–LSTM](ptbxl_cnn_lstm_report.md). The inventory findings below remain the original audit evidence; the two datasets were not merged.

- Metadata: **21,799 ECG records**, **18,869 patients**; 0 missing patient IDs.
- Input: 12 simultaneous leads, 10 seconds; 1,000 samples per lead at 100 Hz or 5,000 samples per lead at 500 Hz. These are two representations of each ECG.
- SHA256 manifest: **87,203/87,203 files matched**, 0 missing, 0 checksum mismatches, 0 read errors; 3.173 GB read.
- Header/byte checks: 0 unparseable headers, 0 byte-size mismatches, 0 initial-value/channel-checksum errors.
- Metadata coverage: 0 referenced files absent from manifest; 0 waveform/header files outside the database.
- Same-rate exact waveform duplicates: 100 Hz: 0, 500 Hz: 0. No near-duplicate waveform search was performed.
- Constant-channel records: 2 rate-specific files; raw int16 minimum (-32768) values: 0. These are descriptive signal checks, not clinical-quality adjudication.

## Existing official partition metadata

| Partition | Records | Patients |
|---|---:|---:|
| train_folds_1_to_8 | 17,418 | 15,023 |
| validation_fold_9 | 2,183 | 1,942 |
| test_fold_10 | 2,198 | 1,904 |

Patients appearing in multiple official folds: **0**. Patient overlap between every train/validation/test pair: **[0, 0, 0]**.

## Local diagnostic superclass labels

| Superclass | ECG records containing label |
|---|---:|
| NORM | 9,514 |
| MI | 5,469 |
| STTC | 5,235 |
| CD | 4,898 |
| HYP | 2,649 |

Count each diagnostic superclass at most once per ECG from all present scp_codes keys where local scp_statements.diagnostic==1; no likelihood threshold. Labels are multilabel and counts overlap.
Records with multiple diagnostic superclasses: **5,144**; without a mapped diagnostic superclass: **411**.

These labels differ from the four image folders (Normal, MI, Abnormal, History_MI). A future waveform experiment needs a separate prespecified task, preprocessing, and evaluation protocol. Keep the existing patient-based folds and never treat both sampling rates as separate patients.

The complete SCP-code frequency table, input dimensions, exact-duplicate groups, anomaly lists, and source checksums are saved in `data/audit/ptbxl_inventory.json`. This audit verifies consistency with the local shipped manifest; it does not independently authenticate that manifest against an online source.

Reproduce after explicitly archiving the existing audit outputs:

```bash
/opt/anaconda3/bin/python -m src.audit_ptbxl
```
