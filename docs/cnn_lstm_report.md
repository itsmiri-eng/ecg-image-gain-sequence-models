# CNN–LSTM experiment — completed 24 September 2026

The requested CNN–LSTM was implemented, executed and verified on the same frozen ECG images as LDA and CNN. Its test accuracy is **71.43%**, balanced accuracy **63.15%**, macro F1 **0.6020**, weighted F1 **0.7168**, and macro one-vs-rest ROC-AUC **0.8944**. The run ID is `cnn_lstm_image_v1_seed42`.

## What the model does

The model receives the unchanged cropped grayscale image, with shape 1×128×256. Three convolution blocks produce a 64×8×16 feature map. Pooling retains two coarse vertical bands and 16 horizontal positions. Each left-to-right position supplies 128 features to a single-layer, unidirectional LSTM with 32 hidden units. The classifier averages its 16 outputs, applies dropout and predicts one of four supplied folder labels.

This is an **image classifier with a spatial sequence of learned features**. It does not extract voltage-versus-time signals or establish physiological timing. It has 44,532 parameters, uses no pretrained weights and introduces no augmentation. The original crop's known waveform clipping, aspect distortion and low resolution remain limitations shared by the compared inputs.

## Executed protocol

The configuration and source hashes were saved before fitting. One configuration and seed 42 were used, with no hyperparameter search. The 491 images retain their original 295/98/98 training/validation/test assignments and class order. Normalization and inverse-frequency class weights were calculated from training data alone.

Training used CPU, batch size 16, AdamW with learning rate 0.001 and weight decay 0.0001, dropout 0.25, and gradient clipping at norm 1.0. The model completed all 30 allowed epochs; the patience-7 rule did not stop it early. **Epoch 27** was selected by highest validation macro F1, then balanced accuracy, then lowest weighted loss, retaining the earlier checkpoint for exact ties. Training took **48.19 seconds** and the runner took **51.05 seconds**, excluding interpreter startup and the separate verification/reporting steps.

The original CNN selected minimum weighted validation loss. Pooling/readout, gradient clipping and the selection rule also differ in this new experiment. It is therefore a comparison of complete pipelines, not an isolated causal test of adding an LSTM.

## Results on the same previously observed test set

| Model | Accuracy | Balanced accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|
| PCA + LDA | 62.24% | 60.32% | 0.5637 | 0.6423 |
| CNN | 75.51% | 56.86% | 0.5621 | 0.7310 |
| CNN–LSTM | 71.43% | 63.15% | 0.6020 | 0.7168 |

| Supplied class | Test images | LDA correct | CNN correct | CNN–LSTM correct |
|---|---:|---:|---:|---:|
| Normal | 28 | 21 | 28 | 26 |
| MI | 6 | 3 | 1 | 3 |
| Abnormal | 47 | 27 | 41 | 35 |
| History_MI | 17 | 10 | 4 | 6 |

CNN–LSTM correctly classifies 70/98 images compared with CNN's 74/98. It recognises more MI and History_MI examples than CNN, while recognising fewer Normal and Abnormal examples. LDA still recognises more History_MI images than either neural model. The new model's higher macro F1 and balanced accuracy describe a more balanced result on this particular split, not general superiority.

At the selected checkpoint, training and validation macro F1 are **0.6177 and 0.6001**; corresponding accuracies are **76.27% and 73.47%**. Later epochs improve training performance while validation macro F1 declines, supporting the use of the saved validation-selected checkpoint. The short, fluctuating curves and small sample counts do not establish that overfitting or underfitting has been resolved.

Test inference occurred once after configuration and checkpoint selection were frozen. Baseline test results had already been inspected, so these are exploratory comparisons on a previously observed test set. The six MI test examples, supplied unadjudicated labels, unverified printed-ID patient semantics and one-seed design limit inference. No test-guided retuning was performed.

## Verification and saved evidence

- All **21 tests passed**, including sequence ordering, deterministic synthetic fitting, training-only statistics, checkpoint selection, failure logging and test isolation.
- Reloading the saved checkpoint reproduced probabilities for all 295 training examples exactly: maximum difference **0.0**, zero prediction disagreements. Verification did not rerun test inference.
- Saved predictions reproduce all reported train/validation/test metrics and confusion matrices for all three models.
- All **66 immutable baseline artifacts** retain their hashes. The central experiment log preserves the original header and two rows byte-for-byte and appends the third completed experiment. The historical 67-file lock remains unchanged; its full-log hash is checked against the preserved pre-append snapshot.
- No new dataset, relabelling or replacement crop was introduced.

Main artifacts:

- [Three-model comparison](../results/comparisons/cnn_lstm_image_v1_seed42/comparison.md) and [comparison figure](../results/comparisons/cnn_lstm_image_v1_seed42/comparison.png).
- [Training curve](../models/cnn_lstm/cnn_lstm_image_v1_seed42/training_curve.png), checkpoint `models/cnn_lstm/cnn_lstm_image_v1_seed42/best.pt`, and epoch history beside it.
- Full metrics, predictions, confusion matrix, configuration, input/source hashes and verification under `results/cnn_lstm/cnn_lstm_image_v1_seed42/`.
- Frozen pre-training specification under `experiments/protocols/cnn_lstm_image_v1/`; complete execution log in `experiments/logs/cnn_lstm_image_v1_seed42.log`.

## PTB-XL decision

PTB-XL is a stronger foundation for a larger subsequent study because the local copy contains 21,799 recordings from 18,869 explicitly identified patients, with documented annotations and official patient folds. The previously claimed 600 generated images and their generator are not present in the inspected workspace; the original waveforms are present and readable. See [PTB-XL migration assessment](ptbxl_migration_assessment.md) for the evidence and official references.

If the research target remains **paper ECG image classification**, generate a separate, versioned image cohort from PTB-XL waveforms, inspect development renderings before bulk generation, and preserve its patient folds. Use an explicit label protocol: the five diagnostic superclasses overlap and are not equivalent to our current four folders. Do not print diagnosis text into classifier inputs or count multiple renderings as independent patients. Synthetic-image performance still needs evaluation on suitable real images.

If the target is **temporal waveform classification**, native 100-Hz PTB-XL signals offer a direct time axis for a 1D CNN–LSTM without rendering or digitization. That is a different experiment and input modality. No PTB-XL model was trained in this run, and its future results must not be presented as a same-task accuracy comparison with the four-class Mendeley experiment. LSM and the remembered convolution term remain unconfirmed.

## Reproduce

From the project root, using a new run ID to protect completed evidence:

```bash
/opt/anaconda3/bin/python -m src.verify
/opt/anaconda3/bin/python -m src.baseline_preservation
/opt/anaconda3/bin/python -m unittest discover -s tests -v
/opt/anaconda3/bin/python -m src.run_cnn_lstm --run-id cnn_lstm_image_reproduction_seed42
/opt/anaconda3/bin/python -m src.compare_cnn_lstm --cnn-lstm-id cnn_lstm_image_reproduction_seed42
```

The runner defaults to `experiments/configs/cnn_lstm_image_v1.json`. Reproduction of unchanged settings is distinct from tuning against the observed test set. Do not overwrite the completed run or use repeated test scores to choose settings.
