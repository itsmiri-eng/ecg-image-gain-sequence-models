# External test: georgia

Eligible labels: STTC, CD, HYP. Frozen plan: docs/external_test_plan.md.

## H1 and H2 (hierarchical seed x record bootstrap, Holm within family)

| Family | Comparison | Metric | Difference [95% CI] | Holm p |
|---|---|---|---|---:|
| H1 | cnn_pool_wide - cnn_pool | macro AUROC | +0.0478 [+0.0309, +0.0644] | 0.0000 |
| H1 | cnn_lstm - cnn_pool | macro AUROC | +0.0711 [+0.0607, +0.0809] | 0.0000 |
| H1 | temporal_conv - cnn_pool | macro AUROC | +0.0551 [+0.0466, +0.0631] | 0.0000 |
| H1 | hybrid_vit - cnn_pool | macro AUROC | +0.0777 [+0.0711, +0.0843] | 0.0000 |
| H1 | cnn_lstm - cnn_pool_wide | macro AUROC | +0.0233 [+0.0087, +0.0382] | 0.0000 |
| H1 | temporal_conv - cnn_pool_wide | macro AUROC | +0.0073 [-0.0103, +0.0249] | 0.4460 |
| H1 | hybrid_vit - cnn_pool_wide | macro AUROC | +0.0299 [+0.0168, +0.0441] | 0.0000 |
| H1 | temporal_conv - cnn_lstm | macro AUROC | -0.0160 [-0.0239, -0.0084] | 0.0000 |
| H1 | hybrid_vit - cnn_lstm | macro AUROC | +0.0066 [-0.0033, +0.0170] | 0.4460 |
| H1 | hybrid_vit - temporal_conv | macro AUROC | +0.0226 [+0.0132, +0.0329] | 0.0000 |
| H2 | cnn_pool: fixed - adaptive | macro AUROC | +0.0550 [+0.0451, +0.0645] | 0.0000 |
| H2 | cnn_pool: fixed - adaptive | HYP AUROC | +0.1287 [+0.1091, +0.1498] | 0.0000 |
| H2 | cnn_lstm: fixed - adaptive | macro AUROC | +0.0152 [+0.0065, +0.0238] | 0.0000 |
| H2 | cnn_lstm: fixed - adaptive | HYP AUROC | +0.0273 [+0.0111, +0.0455] | 0.0000 |
| H2 | temporal_conv: fixed - adaptive | macro AUROC | +0.0307 [+0.0225, +0.0388] | 0.0000 |
| H2 | temporal_conv: fixed - adaptive | HYP AUROC | +0.0957 [+0.0782, +0.1168] | 0.0000 |
| H2 | hybrid_vit: fixed - adaptive | macro AUROC | +0.0076 [+0.0038, +0.0114] | 0.0000 |
| H2 | hybrid_vit: fixed - adaptive | HYP AUROC | +0.0181 [+0.0014, +0.0346] | 0.0300 |

## Per run (H3: external minus PTB-XL test, same labels)

| Run | Page | Seed | External macro AUROC | PTB-XL test | Difference |
|---|---|---:|---:|---:|---:|
| ptbxl_cnn_lstm_image_v1_seed42 | adaptive | 42 | 0.7102 | 0.8691 | -0.1589 |
| ptbxl_hybrid_vit_image_v1_seed42 | adaptive | 42 | 0.7331 | 0.8863 | -0.1532 |
| ptbxl_spatiotemporal_image_v1_seed42 | adaptive | 42 | 0.7056 | 0.8715 | -0.1659 |
| ptbxl_v2_cnn_lstm_seed43 | adaptive | 43 | 0.7332 | 0.8813 | -0.1482 |
| ptbxl_v2_cnn_lstm_seed44 | adaptive | 44 | 0.7238 | 0.8828 | -0.1590 |
| ptbxl_v2_cnn_lstm_seed45 | adaptive | 45 | 0.7225 | 0.8669 | -0.1444 |
| ptbxl_v2_cnn_lstm_seed46 | adaptive | 46 | 0.7264 | 0.8775 | -0.1511 |
| ptbxl_v2_cnn_pool_seed42 | adaptive | 42 | 0.6556 | 0.8257 | -0.1701 |
| ptbxl_v2_cnn_pool_seed43 | adaptive | 43 | 0.6495 | 0.8216 | -0.1721 |
| ptbxl_v2_cnn_pool_seed44 | adaptive | 44 | 0.6459 | 0.8177 | -0.1718 |
| ptbxl_v2_cnn_pool_seed45 | adaptive | 45 | 0.6499 | 0.8239 | -0.1740 |
| ptbxl_v2_cnn_pool_seed46 | adaptive | 46 | 0.6597 | 0.8261 | -0.1664 |
| ptbxl_v2_cnn_pool_wide_seed42 | adaptive | 42 | 0.6798 | 0.8553 | -0.1754 |
| ptbxl_v2_cnn_pool_wide_seed43 | adaptive | 43 | 0.7016 | 0.8640 | -0.1624 |
| ptbxl_v2_cnn_pool_wide_seed44 | adaptive | 44 | 0.7178 | 0.8673 | -0.1494 |
| ptbxl_v2_cnn_pool_wide_seed45 | adaptive | 45 | 0.6781 | 0.8525 | -0.1744 |
| ptbxl_v2_cnn_pool_wide_seed46 | adaptive | 46 | 0.7221 | 0.8712 | -0.1491 |
| ptbxl_v2_fg_cnn_lstm_seed42 | fixed | 42 | 0.7401 | 0.8840 | -0.1439 |
| ptbxl_v2_fg_cnn_lstm_seed43 | fixed | 43 | 0.7360 | 0.8870 | -0.1510 |
| ptbxl_v2_fg_cnn_lstm_seed44 | fixed | 44 | 0.7324 | 0.8878 | -0.1554 |
| ptbxl_v2_fg_cnn_lstm_seed45 | fixed | 45 | 0.7400 | 0.8916 | -0.1516 |
| ptbxl_v2_fg_cnn_lstm_seed46 | fixed | 46 | 0.7435 | 0.8926 | -0.1491 |
| ptbxl_v2_fg_cnn_pool_seed42 | fixed | 42 | 0.6965 | 0.8531 | -0.1566 |
| ptbxl_v2_fg_cnn_pool_seed43 | fixed | 43 | 0.7098 | 0.8683 | -0.1585 |
| ptbxl_v2_fg_cnn_pool_seed44 | fixed | 44 | 0.7137 | 0.8636 | -0.1498 |
| ptbxl_v2_fg_cnn_pool_seed45 | fixed | 45 | 0.7079 | 0.8559 | -0.1480 |
| ptbxl_v2_fg_cnn_pool_seed46 | fixed | 46 | 0.7078 | 0.8643 | -0.1565 |
| ptbxl_v2_fg_hybrid_vit_seed42 | fixed | 42 | 0.7406 | 0.8998 | -0.1592 |
| ptbxl_v2_fg_hybrid_vit_seed43 | fixed | 43 | 0.7337 | 0.8813 | -0.1475 |
| ptbxl_v2_fg_hybrid_vit_seed44 | fixed | 44 | 0.7343 | 0.8915 | -0.1572 |
| ptbxl_v2_fg_hybrid_vit_seed45 | fixed | 45 | 0.7311 | 0.8887 | -0.1576 |
| ptbxl_v2_fg_hybrid_vit_seed46 | fixed | 46 | 0.7474 | 0.8875 | -0.1401 |
| ptbxl_v2_fg_lda_seed42 | fixed | 42 | 0.6597 | 0.7852 | -0.1255 |
| ptbxl_v2_fg_temporal_conv_seed42 | fixed | 42 | 0.7421 | 0.8915 | -0.1494 |
| ptbxl_v2_fg_temporal_conv_seed43 | fixed | 43 | 0.7346 | 0.8892 | -0.1546 |
| ptbxl_v2_fg_temporal_conv_seed44 | fixed | 44 | 0.7403 | 0.8935 | -0.1532 |
| ptbxl_v2_fg_temporal_conv_seed45 | fixed | 45 | 0.7288 | 0.8898 | -0.1610 |
| ptbxl_v2_fg_temporal_conv_seed46 | fixed | 46 | 0.7439 | 0.8953 | -0.1514 |
| ptbxl_v2_hybrid_vit_seed43 | adaptive | 43 | 0.7255 | 0.8867 | -0.1611 |
| ptbxl_v2_hybrid_vit_seed44 | adaptive | 44 | 0.7302 | 0.8835 | -0.1534 |
| ptbxl_v2_hybrid_vit_seed45 | adaptive | 45 | 0.7188 | 0.8797 | -0.1609 |
| ptbxl_v2_hybrid_vit_seed46 | adaptive | 46 | 0.7413 | 0.8830 | -0.1418 |
| ptbxl_v2_lda_seed42 | adaptive | 42 | 0.6652 | 0.7859 | -0.1207 |
| ptbxl_v2_lda_wide_seed42 | adaptive | 42 | 0.6670 | 0.7915 | -0.1246 |
| ptbxl_v2_signal_resnet1d_seed42 | signal | 42 | 0.7873 | 0.9216 | -0.1343 |
| ptbxl_v2_signal_resnet1d_seed43 | signal | 43 | 0.7876 | 0.9206 | -0.1330 |
| ptbxl_v2_signal_resnet1d_seed44 | signal | 44 | 0.7896 | 0.9196 | -0.1300 |
| ptbxl_v2_svm_seed42 | adaptive | 42 | 0.6671 | 0.7909 | -0.1239 |
| ptbxl_v2_temporal_conv_seed43 | adaptive | 43 | 0.7062 | 0.8699 | -0.1637 |
| ptbxl_v2_temporal_conv_seed44 | adaptive | 44 | 0.7105 | 0.8760 | -0.1655 |
| ptbxl_v2_temporal_conv_seed45 | adaptive | 45 | 0.7115 | 0.8744 | -0.1629 |
| ptbxl_v2_temporal_conv_seed46 | adaptive | 46 | 0.7023 | 0.8678 | -0.1655 |
