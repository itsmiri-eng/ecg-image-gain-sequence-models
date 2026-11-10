# Executed LDA vs CNN comparison

Both models used the same frozen cropped-image dataset and identity-grouped split. Metrics below are recomputed from saved test predictions; no model was refitted to produce this comparison.

| Model | Accuracy | Balanced accuracy | Macro F1 | Weighted F1 | Notes |
|---|---:|---:|---:|---:|---|
| LDA | 0.6224 | 0.6032 | 0.5637 | 0.6423 | Train-only PCA(32), uniform-prior shrinkage LDA |
| CNN | 0.7551 | 0.5686 | 0.5621 | 0.7310 | 40,212 parameters; weighted CE; validation-only early stopping |

CNN minus LDA: accuracy +0.1327, balanced accuracy -0.0346, macro F1 -0.0016, weighted F1 +0.0887.

These are observed differences on one small split, not evidence of general clinical superiority. Printed IDs are unverified patient proxies, and near-duplicate screening is not proof of independence.

| Class | Test n | LDA recall | CNN recall | LDA F1 | CNN F1 | F1 change |
|---|---:|---:|---:|---:|---:|---:|
| Normal | 28 | 0.7500 | 1.0000 | 0.7000 | 0.7467 | +0.0467 |
| MI | 6 | 0.5000 | 0.1667 | 0.4000 | 0.2500 | -0.1500 |
| Abnormal | 47 | 0.5745 | 0.8723 | 0.7105 | 0.9318 | +0.2213 |
| History_MI | 17 | 0.5882 | 0.2353 | 0.4444 | 0.3200 | -0.1244 |

## Generalization checks

| Model | Train accuracy | Validation accuracy | Test accuracy | Train macro F1 | Validation macro F1 | Test macro F1 |
|---|---:|---:|---:|---:|---:|---:|
| LDA | 0.7220 | 0.6224 | 0.6224 | 0.6740 | 0.5296 | 0.5637 |
| CNN | 0.7220 | 0.7143 | 0.7551 | 0.4360 | 0.4607 | 0.5621 |

CNN selected epoch 25 of 30 by minimum weighted validation loss. See its saved history/curve for the training and validation trajectories.

LDA tests linear discriminative structure after an unsupervised train-only 32-component PCA projection. Good or poor performance here does not establish whether raw ECG physiology is linearly separable: crop resolution, PCA information loss, source artifacts, class overlap, and small support all matter.

Both models may use page-layout/grid artifacts remaining within the trace crop. The 128×256 image reduction is a computational baseline, not a claim of clinically lossless digitization. Neither model receives calibrated physical waveforms.

Resolve the supervisor's LSM and remembered convolution term before implementing those steps. Before CNN+LSTM, validate lead/time extraction and obtain more independent minority-class data.
