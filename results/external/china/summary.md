# External test: china

Eligible labels: MI, STTC, CD, HYP. Frozen plan: docs/external_test_plan.md.

## H1 and H2 (hierarchical seed x record bootstrap, Holm within family)

| Family | Comparison | Metric | Difference [95% CI] | Holm p |
|---|---|---|---|---:|
| H1 | cnn_pool_wide - cnn_pool | macro AUROC | +0.0599 [+0.0453, +0.0774] | 0.0020 |
| H1 | cnn_lstm - cnn_pool | macro AUROC | +0.0827 [+0.0674, +0.1002] | 0.0020 |
| H1 | temporal_conv - cnn_pool | macro AUROC | +0.0940 [+0.0817, +0.1069] | 0.0020 |
| H1 | hybrid_vit - cnn_pool | macro AUROC | +0.0986 [+0.0849, +0.1149] | 0.0020 |
| H1 | cnn_lstm - cnn_pool_wide | macro AUROC | +0.0228 [+0.0110, +0.0355] | 0.0020 |
| H1 | temporal_conv - cnn_pool_wide | macro AUROC | +0.0341 [+0.0248, +0.0431] | 0.0020 |
| H1 | hybrid_vit - cnn_pool_wide | macro AUROC | +0.0386 [+0.0290, +0.0487] | 0.0020 |
| H1 | temporal_conv - cnn_lstm | macro AUROC | +0.0113 [+0.0002, +0.0236] | 0.0932 |
| H1 | hybrid_vit - cnn_lstm | macro AUROC | +0.0159 [+0.0094, +0.0227] | 0.0020 |
| H1 | hybrid_vit - temporal_conv | macro AUROC | +0.0045 [-0.0061, +0.0137] | 0.3818 |
| H2 | cnn_pool: fixed - adaptive | macro AUROC | +0.0511 [+0.0353, +0.0718] | 0.0016 |
| H2 | cnn_pool: fixed - adaptive | HYP AUROC | +0.1124 [+0.0876, +0.1399] | 0.0016 |
| H2 | cnn_lstm: fixed - adaptive | macro AUROC | +0.0176 [+0.0052, +0.0310] | 0.0160 |
| H2 | cnn_lstm: fixed - adaptive | HYP AUROC | +0.0348 [+0.0237, +0.0464] | 0.0016 |
| H2 | temporal_conv: fixed - adaptive | macro AUROC | +0.0137 [+0.0044, +0.0227] | 0.0162 |
| H2 | temporal_conv: fixed - adaptive | HYP AUROC | +0.0333 [+0.0232, +0.0448] | 0.0016 |
| H2 | hybrid_vit: fixed - adaptive | macro AUROC | +0.0036 [-0.0044, +0.0121] | 0.3806 |
| H2 | hybrid_vit: fixed - adaptive | HYP AUROC | +0.0138 [+0.0036, +0.0253] | 0.0162 |

## Per run (H3: external minus PTB-XL test, same labels)

| Run | Page | Seed | External macro AUROC | PTB-XL test | Difference |
|---|---|---:|---:|---:|---:|
| ptbxl_cnn_lstm_image_v1_seed42 | adaptive | 42 | 0.7931 | 0.8591 | -0.0660 |
| ptbxl_hybrid_vit_image_v1_seed42 | adaptive | 42 | 0.8157 | 0.8819 | -0.0661 |
| ptbxl_spatiotemporal_image_v1_seed42 | adaptive | 42 | 0.8127 | 0.8704 | -0.0576 |
| ptbxl_v2_cnn_lstm_seed43 | adaptive | 43 | 0.8043 | 0.8711 | -0.0668 |
| ptbxl_v2_cnn_lstm_seed44 | adaptive | 44 | 0.8115 | 0.8770 | -0.0655 |
| ptbxl_v2_cnn_lstm_seed45 | adaptive | 45 | 0.7889 | 0.8613 | -0.0724 |
| ptbxl_v2_cnn_lstm_seed46 | adaptive | 46 | 0.7990 | 0.8628 | -0.0638 |
| ptbxl_v2_cnn_pool_seed42 | adaptive | 42 | 0.7275 | 0.8195 | -0.0920 |
| ptbxl_v2_cnn_pool_seed43 | adaptive | 43 | 0.6949 | 0.8132 | -0.1183 |
| ptbxl_v2_cnn_pool_seed44 | adaptive | 44 | 0.7165 | 0.8108 | -0.0943 |
| ptbxl_v2_cnn_pool_seed45 | adaptive | 45 | 0.7186 | 0.8178 | -0.0992 |
| ptbxl_v2_cnn_pool_seed46 | adaptive | 46 | 0.7258 | 0.8187 | -0.0929 |
| ptbxl_v2_cnn_pool_wide_seed42 | adaptive | 42 | 0.7856 | 0.8550 | -0.0694 |
| ptbxl_v2_cnn_pool_wide_seed43 | adaptive | 43 | 0.7856 | 0.8578 | -0.0721 |
| ptbxl_v2_cnn_pool_wide_seed44 | adaptive | 44 | 0.7708 | 0.8529 | -0.0820 |
| ptbxl_v2_cnn_pool_wide_seed45 | adaptive | 45 | 0.7760 | 0.8463 | -0.0702 |
| ptbxl_v2_cnn_pool_wide_seed46 | adaptive | 46 | 0.7647 | 0.8624 | -0.0977 |
| ptbxl_v2_fg_cnn_lstm_seed42 | fixed | 42 | 0.8127 | 0.8744 | -0.0617 |
| ptbxl_v2_fg_cnn_lstm_seed43 | fixed | 43 | 0.8142 | 0.8764 | -0.0622 |
| ptbxl_v2_fg_cnn_lstm_seed44 | fixed | 44 | 0.8118 | 0.8788 | -0.0670 |
| ptbxl_v2_fg_cnn_lstm_seed45 | fixed | 45 | 0.8295 | 0.8872 | -0.0577 |
| ptbxl_v2_fg_cnn_lstm_seed46 | fixed | 46 | 0.8167 | 0.8834 | -0.0667 |
| ptbxl_v2_fg_cnn_pool_seed42 | fixed | 42 | 0.7593 | 0.8412 | -0.0818 |
| ptbxl_v2_fg_cnn_pool_seed43 | fixed | 43 | 0.7830 | 0.8571 | -0.0741 |
| ptbxl_v2_fg_cnn_pool_seed44 | fixed | 44 | 0.7556 | 0.8502 | -0.0946 |
| ptbxl_v2_fg_cnn_pool_seed45 | fixed | 45 | 0.7672 | 0.8440 | -0.0768 |
| ptbxl_v2_fg_cnn_pool_seed46 | fixed | 46 | 0.7737 | 0.8512 | -0.0775 |
| ptbxl_v2_fg_cnn_pool_wide_seed42 | fixed | 42 | 0.8128 | 0.8739 | -0.0611 |
| ptbxl_v2_fg_cnn_pool_wide_seed43 | fixed | 43 | 0.7999 | 0.8655 | -0.0656 |
| ptbxl_v2_fg_cnn_pool_wide_seed44 | fixed | 44 | 0.8091 | 0.8711 | -0.0620 |
| ptbxl_v2_fg_cnn_pool_wide_seed45 | fixed | 45 | 0.8166 | 0.8784 | -0.0618 |
| ptbxl_v2_fg_cnn_pool_wide_seed46 | fixed | 46 | 0.8089 | 0.8809 | -0.0721 |
| ptbxl_v2_fg_hybrid_vit_seed42 | fixed | 42 | 0.8146 | 0.8907 | -0.0761 |
| ptbxl_v2_fg_hybrid_vit_seed43 | fixed | 43 | 0.8155 | 0.8727 | -0.0572 |
| ptbxl_v2_fg_hybrid_vit_seed44 | fixed | 44 | 0.8245 | 0.8856 | -0.0611 |
| ptbxl_v2_fg_hybrid_vit_seed45 | fixed | 45 | 0.8229 | 0.8860 | -0.0631 |
| ptbxl_v2_fg_hybrid_vit_seed46 | fixed | 46 | 0.8168 | 0.8798 | -0.0629 |
| ptbxl_v2_fg_lda_seed42 | fixed | 42 | 0.7517 | 0.7768 | -0.0251 |
| ptbxl_v2_fg_temporal_conv_seed42 | fixed | 42 | 0.8345 | 0.8876 | -0.0531 |
| ptbxl_v2_fg_temporal_conv_seed43 | fixed | 43 | 0.8129 | 0.8848 | -0.0719 |
| ptbxl_v2_fg_temporal_conv_seed44 | fixed | 44 | 0.8299 | 0.8862 | -0.0564 |
| ptbxl_v2_fg_temporal_conv_seed45 | fixed | 45 | 0.8196 | 0.8863 | -0.0667 |
| ptbxl_v2_fg_temporal_conv_seed46 | fixed | 46 | 0.8250 | 0.8880 | -0.0630 |
| ptbxl_v2_hybrid_vit_seed43 | adaptive | 43 | 0.8200 | 0.8804 | -0.0605 |
| ptbxl_v2_hybrid_vit_seed44 | adaptive | 44 | 0.8195 | 0.8773 | -0.0578 |
| ptbxl_v2_hybrid_vit_seed45 | adaptive | 45 | 0.8062 | 0.8675 | -0.0613 |
| ptbxl_v2_hybrid_vit_seed46 | adaptive | 46 | 0.8147 | 0.8787 | -0.0640 |
| ptbxl_v2_lda_seed42 | adaptive | 42 | 0.7343 | 0.7822 | -0.0479 |
| ptbxl_v2_lda_wide_seed42 | adaptive | 42 | 0.7324 | 0.7867 | -0.0543 |
| ptbxl_v2_signal_resnet1d_seed42 | signal | 42 | 0.8523 | 0.9248 | -0.0725 |
| ptbxl_v2_signal_resnet1d_seed43 | signal | 43 | 0.8537 | 0.9235 | -0.0698 |
| ptbxl_v2_signal_resnet1d_seed44 | signal | 44 | 0.8610 | 0.9231 | -0.0621 |
| ptbxl_v2_signal_resnet1d_seed45 | signal | 45 | 0.8506 | 0.9202 | -0.0695 |
| ptbxl_v2_signal_resnet1d_seed46 | signal | 46 | 0.8593 | 0.9231 | -0.0638 |
| ptbxl_v2_svm_seed42 | adaptive | 42 | 0.7296 | 0.7856 | -0.0560 |
| ptbxl_v2_temporal_conv_seed43 | adaptive | 43 | 0.8061 | 0.8671 | -0.0611 |
| ptbxl_v2_temporal_conv_seed44 | adaptive | 44 | 0.8104 | 0.8683 | -0.0579 |
| ptbxl_v2_temporal_conv_seed45 | adaptive | 45 | 0.8202 | 0.8727 | -0.0525 |
| ptbxl_v2_temporal_conv_seed46 | adaptive | 46 | 0.8039 | 0.8643 | -0.0605 |
