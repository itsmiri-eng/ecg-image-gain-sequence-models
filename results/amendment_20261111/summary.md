# Amendment of 11 Nov 2026: results (2026-11-15T17:31 UTC)

Protocol: docs/protocol_amendment_2026-11-11.md. PTB-XL fold 10 was observed before: exploratory sensitivity analyses.

## B. ImageNet-pretrained ResNet-18 (seeds 42-44)

| Cohort | Pages | Macro AUROC (mean ± SD) | HYP AUROC (mean) | Selected epochs |
|---|---|---:|---:|---|
| PTB-XL test | adaptive | 0.9207 ± 0.0009 | 0.8854 | 8 7 8 |
| PTB-XL test | fixed | 0.9289 ± 0.0021 | 0.9073 | 7 10 9 |
| georgia | adaptive | 0.7734 ± 0.0011 | 0.7220 |    |
| georgia | fixed | 0.7786 ± 0.0037 | 0.7330 |    |
| china | adaptive | 0.8482 ± 0.0029 | 0.8327 |    |
| china | fixed | 0.8527 ± 0.0021 | 0.8446 |    |

Fixed - adaptive gain. PTB-XL: seed x patient bootstrap, Holm over 2. External: seed x record bootstrap, Holm over the cohort x metric comparisons available.

| Cohort | Metric | Difference [95% CI] | Holm p |
|---|---|---|---:|
| PTB-XL test | macro AUROC | +0.0081 [+0.0035, +0.0128] | 0.0012 |
| PTB-XL test | HYP AUROC | +0.0219 [+0.0086, +0.0352] | 0.0014 |
| georgia | macro AUROC (eligible labels) | +0.0052 [-0.0002, +0.0098] | 0.1752 |
| georgia | HYP AUROC | +0.0109 [+0.0020, +0.0193] | 0.0760 |
| china | macro AUROC (eligible labels) | +0.0045 [-0.0016, +0.0110] | 0.3016 |
| china | HYP AUROC | +0.0119 [-0.0066, +0.0303] | 0.3016 |

## A. 40-epoch sensitivity (seeds 42-44, adaptive gain)

| Head | Epochs | Test macro AUROC (mean ± SD) | Validation | GPU-only test (n) | Selected epochs | Epochs run |
|---|---:|---:|---:|---:|---|---|
| cnn_pool | 20 | 0.8291 ± 0.0047 | 0.8410 | 0.8264 (2) | 19 20 19 | 20 20 20 |
| cnn_pool | 40 | 0.8448 ± 0.0031 | 0.8584 | 0.8448 (3) | 33 37 36 | 37 40 40 |
| cnn_pool_wide | 20 | 0.8650 ± 0.0027 | 0.8758 | 0.8650 (3) | 20 20 18 | 20 20 20 |
| cnn_pool_wide | 40 | 0.8836 ± 0.0007 | 0.8917 | 0.8836 (3) | 37 31 35 | 40 35 39 |
| cnn_lstm | 20 | 0.8789 ± 0.0084 | 0.8925 | 0.8835 (2) | 15 16 18 | 19 20 20 |
| cnn_lstm | 40 | 0.8910 ± 0.0033 | 0.9029 | 0.8910 (3) | 27 26 28 | 31 30 32 |
| temporal_conv | 20 | 0.8794 ± 0.0008 | 0.8875 | 0.8789 (2) | 16 19 20 | 20 20 20 |
| temporal_conv | 40 | 0.8914 ± 0.0024 | 0.8993 | 0.8914 (3) | 29 35 30 | 33 39 34 |
| hybrid_vit | 20 | 0.8880 ± 0.0023 | 0.8980 | 0.8869 (2) | 19 18 13 | 20 20 17 |
| hybrid_vit | 40 | 0.8912 ± 0.0055 | 0.9001 | 0.8912 (3) | 15 30 21 | 19 34 25 |

Hierarchical seed x patient bootstrap on the PTB-XL test fold (2,000 resamples), Holm within family.

| Family | Comparison | Difference [95% CI] | Holm p |
|---|---|---|---:|
| 40 - 20 epochs | cnn_pool | +0.0158 [+0.0109, +0.0199] | 0.0010 |
| 40 - 20 epochs | cnn_pool_wide | +0.0186 [+0.0148, +0.0225] | 0.0010 |
| 40 - 20 epochs | cnn_lstm | +0.0121 [+0.0049, +0.0213] | 0.0010 |
| 40 - 20 epochs | temporal_conv | +0.0120 [+0.0078, +0.0160] | 0.0010 |
| 40 - 20 epochs | hybrid_vit | +0.0032 [-0.0030, +0.0097] | 0.3124 |
| head pairs, 40 epochs | cnn_pool_wide - cnn_pool | +0.0388 [+0.0324, +0.0455] | 0.0020 |
| head pairs, 40 epochs | cnn_lstm - cnn_pool | +0.0462 [+0.0382, +0.0550] | 0.0020 |
| head pairs, 40 epochs | temporal_conv - cnn_pool | +0.0466 [+0.0395, +0.0535] | 0.0020 |
| head pairs, 40 epochs | hybrid_vit - cnn_pool | +0.0464 [+0.0381, +0.0552] | 0.0020 |
| head pairs, 40 epochs | cnn_lstm - cnn_pool_wide | +0.0073 [+0.0012, +0.0134] | 0.1100 |
| head pairs, 40 epochs | temporal_conv - cnn_pool_wide | +0.0078 [+0.0029, +0.0127] | 0.0264 |
| head pairs, 40 epochs | hybrid_vit - cnn_pool_wide | +0.0076 [+0.0010, +0.0145] | 0.1100 |
| head pairs, 40 epochs | temporal_conv - cnn_lstm | +0.0004 [-0.0065, +0.0077] | 1.0000 |
| head pairs, 40 epochs | hybrid_vit - cnn_lstm | +0.0003 [-0.0082, +0.0099] | 1.0000 |
| head pairs, 40 epochs | hybrid_vit - temporal_conv | -0.0001 [-0.0058, +0.0056] | 1.0000 |
| head pairs, 20 epochs (same seeds) | cnn_pool_wide - cnn_pool | +0.0360 [+0.0284, +0.0434] | 0.0020 |
| head pairs, 20 epochs (same seeds) | cnn_lstm - cnn_pool | +0.0499 [+0.0359, +0.0631] | 0.0020 |
| head pairs, 20 epochs (same seeds) | temporal_conv - cnn_pool | +0.0503 [+0.0427, +0.0579] | 0.0020 |
| head pairs, 20 epochs (same seeds) | hybrid_vit - cnn_pool | +0.0590 [+0.0509, +0.0672] | 0.0020 |
| head pairs, 20 epochs (same seeds) | cnn_lstm - cnn_pool_wide | +0.0139 [+0.0038, +0.0244] | 0.0240 |
| head pairs, 20 epochs (same seeds) | temporal_conv - cnn_pool_wide | +0.0144 [+0.0087, +0.0201] | 0.0020 |
| head pairs, 20 epochs (same seeds) | hybrid_vit - cnn_pool_wide | +0.0230 [+0.0169, +0.0291] | 0.0020 |
| head pairs, 20 epochs (same seeds) | temporal_conv - cnn_lstm | +0.0005 [-0.0087, +0.0105] | 0.9427 |
| head pairs, 20 epochs (same seeds) | hybrid_vit - cnn_lstm | +0.0091 [-0.0009, +0.0205] | 0.1588 |
| head pairs, 20 epochs (same seeds) | hybrid_vit - temporal_conv | +0.0086 [+0.0036, +0.0139] | 0.0056 |

## C. Fixed-gain training with 40 epochs (seeds 42-44)

| Head | Seed | Selected epoch | Epochs run | Test macro AUROC |
|---|---:|---:|---:|---:|
| cnn_pool | 42 | 40 | 40 | 0.8715 |
| cnn_pool | 43 | 39 | 40 | 0.8826 |
| cnn_pool | 44 | 40 | 40 | 0.8768 |
| cnn_lstm | 42 | 24 | 28 | 0.8895 |
| cnn_lstm | 43 | 27 | 31 | 0.8983 |
| cnn_lstm | 44 | 26 | 30 | 0.8921 |
| temporal_conv | 42 | 11 | 15 | 0.8918 |
| temporal_conv | 43 | 27 | 31 | 0.8986 |
| temporal_conv | 44 | 20 | 24 | 0.8997 |
| hybrid_vit | 42 | 19 | 23 | 0.8998 |
| hybrid_vit | 43 | 27 | 31 | 0.8905 |
| hybrid_vit | 44 | 14 | 18 | 0.8940 |

Hierarchical seed x patient bootstrap on the PTB-XL test fold (2,000 resamples), Holm within family.

| Family | Head | Metric | Difference [95% CI] | Holm p |
|---|---|---|---|---:|
| fixed - adaptive, 40 epochs | cnn_pool | macro AUROC | +0.0322 [+0.0230, +0.0410] | 0.0016 |
| fixed - adaptive, 40 epochs | cnn_pool | HYP AUROC | +0.0604 [+0.0362, +0.0873] | 0.0016 |
| fixed - adaptive, 40 epochs | cnn_lstm | macro AUROC | +0.0023 [-0.0054, +0.0112] | 1.0000 |
| fixed - adaptive, 40 epochs | cnn_lstm | HYP AUROC | +0.0002 [-0.0132, +0.0147] | 1.0000 |
| fixed - adaptive, 40 epochs | temporal_conv | macro AUROC | +0.0053 [-0.0012, +0.0120] | 0.5269 |
| fixed - adaptive, 40 epochs | temporal_conv | HYP AUROC | +0.0229 [+0.0101, +0.0361] | 0.0036 |
| fixed - adaptive, 40 epochs | hybrid_vit | macro AUROC | +0.0035 [-0.0070, +0.0133] | 1.0000 |
| fixed - adaptive, 40 epochs | hybrid_vit | HYP AUROC | +0.0054 [-0.0128, +0.0245] | 1.0000 |
| fixed gain, 40 - 20 epochs | cnn_pool | macro AUROC | +0.0164 [+0.0132, +0.0197] | 0.0008 |
| fixed gain, 40 - 20 epochs | cnn_lstm | macro AUROC | +0.0076 [+0.0031, +0.0129] | 0.0018 |
| fixed gain, 40 - 20 epochs | temporal_conv | macro AUROC | +0.0021 [-0.0034, +0.0068] | 0.4562 |
| fixed gain, 40 - 20 epochs | hybrid_vit | macro AUROC | +0.0035 [-0.0010, +0.0090] | 0.3464 |
| fixed - adaptive, 20 epochs (same seeds) | cnn_pool | macro AUROC | +0.0315 [+0.0190, +0.0431] | 0.0016 |
| fixed - adaptive, 20 epochs (same seeds) | cnn_pool | HYP AUROC | +0.0733 [+0.0454, +0.1038] | 0.0016 |
| fixed - adaptive, 20 epochs (same seeds) | cnn_lstm | macro AUROC | +0.0067 [-0.0010, +0.0151] | 0.3608 |
| fixed - adaptive, 20 epochs (same seeds) | cnn_lstm | HYP AUROC | +0.0107 [-0.0058, +0.0291] | 0.6485 |
| fixed - adaptive, 20 epochs (same seeds) | temporal_conv | macro AUROC | +0.0152 [+0.0105, +0.0202] | 0.0016 |
| fixed - adaptive, 20 epochs (same seeds) | temporal_conv | HYP AUROC | +0.0403 [+0.0257, +0.0563] | 0.0016 |
| fixed - adaptive, 20 epochs (same seeds) | hybrid_vit | macro AUROC | +0.0032 [-0.0065, +0.0120] | 0.6485 |
| fixed - adaptive, 20 epochs (same seeds) | hybrid_vit | HYP AUROC | +0.0083 [-0.0066, +0.0235] | 0.6485 |

