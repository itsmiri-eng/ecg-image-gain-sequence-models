# Amendment of 11 Nov 2026: results (2026-11-11T08:33 UTC)

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
| PTB-XL test | macro AUROC | +0.0081 [+0.0036, +0.0127] | 0.0000 |
| PTB-XL test | HYP AUROC | +0.0219 [+0.0088, +0.0348] | 0.0010 |
| georgia | macro AUROC (eligible labels) | +0.0052 [+0.0001, +0.0100] | 0.1470 |
| georgia | HYP AUROC | +0.0109 [+0.0019, +0.0195] | 0.0840 |
| china | macro AUROC (eligible labels) | +0.0045 [-0.0015, +0.0111] | 0.3020 |
| china | HYP AUROC | +0.0119 [-0.0067, +0.0298] | 0.3020 |

## A. 40-epoch sensitivity (seeds 42-44, adaptive gain)

Incomplete: 40-epoch runs missing for cnn_pool:42, cnn_pool:43, cnn_pool:44, cnn_pool_wide:42, cnn_pool_wide:43, cnn_pool_wide:44, cnn_lstm:42, cnn_lstm:43, cnn_lstm:44, temporal_conv:42, temporal_conv:43, temporal_conv:44, hybrid_vit:42, hybrid_vit:43, hybrid_vit:44. Analysis postponed.

