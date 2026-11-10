# Executed image-model comparison

Three completed models use the same 491-image dataset and 295/98/98 grouped split. Values were reconciled with saved predictions; this report performs no inference.

## Previously observed test set — exploratory comparison

| Model | Accuracy | Balanced accuracy | Macro F1 | Weighted F1 | Macro ROC-AUC |
|---|---:|---:|---:|---:|---:|
| LDA | 62.24% | 60.32% | 0.5637 | 0.6423 | 0.8771 |
| CNN | 75.51% | 56.86% | 0.5621 | 0.7310 | 0.9008 |
| CNN–LSTM | 71.43% | 63.15% | 0.6020 | 0.7168 | 0.8944 |

CNN–LSTM minus CNN: accuracy -4.08 percentage points; macro F1 +0.0399; balanced accuracy +6.30 percentage points.

| Class | Test n | LDA correct | CNN correct | CNN–LSTM correct | CNN–LSTM recall |
|---|---:|---:|---:|---:|---:|
| Normal | 28 | 21 | 28 | 26 | 92.86% |
| MI | 6 | 3 | 1 | 3 | 50.00% |
| Abnormal | 47 | 27 | 41 | 35 | 74.47% |
| History_MI | 17 | 10 | 4 | 6 | 35.29% |

## Development and fitting

| Model | Train accuracy | Validation accuracy | Train macro F1 | Validation macro F1 |
|---|---:|---:|---:|---:|
| LDA | 72.20% | 62.24% | 0.6740 | 0.5296 |
| CNN | 72.20% | 71.43% | 0.4360 | 0.4607 |
| CNN–LSTM | 76.27% | 73.47% | 0.6177 | 0.6001 |

CNN–LSTM: 44,532 parameters, 30 epochs run, selected epoch 27, 48.19 seconds training on cpu.

CNN–LSTM uses the CNN's three convolution blocks followed by coarse vertical pooling and 16 left-to-right spatial positions. A one-layer LSTM with 32 hidden units processes these positions; the classifier uses their mean output. This sequence is image geometry, not calibrated physiological time. No ECG waveform is digitized.

The new checkpoint was selected by validation macro F1, with balanced accuracy then lower weighted loss as tie-breaks. The original CNN selected minimum weighted validation loss. Architecture, pooling/readout, gradient clipping and selection policy differ: this is a comparison of complete pipelines, not an isolated test of adding an LSTM.

One seed and one configuration were run; there was no test-guided retuning. The test set had already been inspected during the baseline project, so this is exploratory evidence. Only six test images are MI. No significance or clinical-superiority claim is supported.

Inputs retain the original low resolution, aspect distortion and known crop clipping. Labels are supplied folder labels, and grouping IDs are not verified patients. These limitations can affect every image model.

PTB-XL is a separate proposed dataset transition. No PTB-XL results are included in this comparison; its patient folds, input representation and multilabel targets require a separately defined experiment.
