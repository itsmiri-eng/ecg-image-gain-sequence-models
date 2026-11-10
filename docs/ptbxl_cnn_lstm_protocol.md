# PTB-XL image CNN–LSTM protocol

This protocol was fixed before training `ptbxl_cnn_lstm_image_v1_seed42`. The numerical configuration is [saved here](</Users/miriamistanboli/Downloads/ecg swiss/experiments/configs/ptbxl_cnn_lstm_image_v1.json>), with [pretraining checks](</Users/miriamistanboli/Downloads/ecg swiss/experiments/protocols/ptbxl_cnn_lstm_image_v1/pretraining_checks.json>). These documents specify the experiment; they do not claim that training has completed. The completed run and independently verified report will be the authority for results.

The input is the separately generated PTB-XL image cohort. Classification uses five overlapping diagnostic targets, NORM, MI, STTC, CD and HYP. The official patient folds are retained: 17,084 eligible records in training, 2,146 in validation and 2,158 in test. The 411 records without a mapped diagnostic superclass remain outside classification. All present diagnostic SCP keys are mapped without a confidence threshold, as documented in the [dataset card](</Users/miriamistanboli/Downloads/ecg swiss/docs/ptbxl_image_dataset.md>). The [official PTB-XL release](https://physionet.org/content/ptb-xl/1.0.3/) provides the source data and patient-fold recommendation.

## Image representation

The full twelve-row region, source pixels `y=80:2960, x=0:2240`, becomes twelve fixed lead channels. This retains all ten seconds of every lead and each row's calibration pulse. Header and footer text are excluded. Grayscale is inverted to ink intensity; each non-overlapping 2×2 block contributes its maximum ink value. The resulting input has shape **12×120×1120**. This is isotropic reduction by two, with no waveform digitization, per-image normalization, thresholding, filtering, flip or random crop.

A global mean and standard deviation are fitted only to the training inputs and applied to all partitions. Compressed PNG inputs are decoded in batches instead of storing the roughly 32 GiB of uncompressed input pixels in memory. Each source page is checked against its saved encoded and pixel hashes during preparation; each cached input is checked again when loaded.

Nine training-fold examples were visually reviewed after preprocessing, including amplitude extremes and the known flat V5 example. One technical preview, ECG 20187, has no mapped diagnostic label and remains excluded from actual classifier inputs. All twelve rows, full horizontal intervals and calibration pulses remain visible. Maximum-ink pooling thickens thin strokes and grid lines; it does not preserve every small morphological detail. Extremely low page gain already compresses detail in the source rendering. These checks establish technical consistency, not clinical fidelity.

## Model and fitting

The network has **18,025 trainable parameters**. A depthwise convolution first filters the twelve lead channels separately, and pointwise convolutions combine the aligned leads. Three small convolution stages produce 24 feature channels. Pooling retains four vertical bands and 70 horizontal positions, giving a **70×96** feature sequence. A one-layer, unidirectional LSTM with 32 hidden units processes this sequence. Mean sequence output, dropout 0.2 and a linear layer produce five logits.

The sequence includes calibration and margin regions as well as the aligned waveform interval. Its positions are image features, not recovered 500 Hz voltage samples. No numeric patient, diagnosis, demographic, gain or provenance metadata are predictor inputs; calibration remains visible in the image pixels.

Five independent sigmoid probabilities permit overlapping diagnoses. Training uses unweighted mean [binary cross-entropy with logits](https://docs.pytorch.org/docs/2.9/generated/torch.nn.BCEWithLogitsLoss.html), AdamW with initial learning rate 0.001 and weight decay 0.0001, gradient clipping at norm 1.0, and a cosine schedule down to 0.00001 over at most 20 epochs. Batch size is 16, with four CPU threads and no data-loader subprocesses. Seed 42 and deterministic PyTorch algorithms are fixed. No pretrained weights, class weighting or augmentation are used.

The displayed training loss is the record-weighted mean of online batch losses, with weights changing during the epoch. It is not a second full training-set evaluation after every epoch. Validation is evaluated in full after each epoch, with dropout disabled. A final full training-set evaluation is saved for the selected checkpoint.

## Selection and held-out evaluation

Select the strict lexicographic maximum of **validation macro ROC-AUC**, then macro average precision, then negative probability-based binary cross-entropy. Exact ties keep the earliest epoch. Stop after four epochs without a selection improvement, or after epoch 20. This is one configuration and one seed. Synthetic CPU probes compared computational feasibility without any ECG validation or test results.

All binary decisions use a fixed threshold of **0.5**. There is no threshold tuning. After training, freeze the checkpoint, code, configuration, cache and normalization. Recompute validation metrics from the restored checkpoint and replay 32 training examples through a separately reconstructed checkpoint. Only then perform one final model inference on the official test partition. Save its predictions before calculating uncertainty; bootstrap calculations reuse predictions and do not rerun the model.

Primary performance is macro ROC-AUC. Also report macro average precision, macro/micro/weighted F1, per-label precision, sensitivity, specificity and confusion matrices. **Exact-match accuracy** requires all five decisions on a record to be correct. **Label accuracy** is the fraction of the five individual decisions that are correct. These measures must be named explicitly, because they differ from the earlier four-class accuracy.

Use 500 patient-cluster bootstrap resamples, seed 2024, for 95% percentile intervals on the final test predictions. Sampling patients with replacement retains all ECGs from each sampled patient, including multiplicity. These intervals describe test-sampling uncertainty conditional on this fitted model; they exclude variation from training seeds, model development and new clinical environments.

## Preservation and interpretation

The earlier LDA, CNN and spatial CNN–LSTM artifacts are preserved. Append one new model-run record to the central experiment log while keeping its previous three rows byte-for-byte. Its single-label accuracy fields remain blank; the new task's explicitly named metrics are stored with the new run. Do not directly rank five-label PTB-XL scores against the four-class Mendeley results.

This experiment evaluates one controlled image-rendering style. It cannot establish performance on photographs, scans, other ECG layouts or clinical deployment. Annotation uncertainty, source artifacts, adaptive page gain and rasterization loss remain. A successful result is an internally evaluated research baseline; external validation and matched-dataset model comparisons remain separate work.

Reproduction requires a new run ID because completed or failed experiment directories cannot be overwritten:

```bash
cd "/Users/miriamistanboli/Downloads/ecg swiss"
/opt/anaconda3/bin/python -m src.ptbxl_learning_data --workers 4
/opt/anaconda3/bin/python -m src.run_ptbxl_cnn_lstm --run-id ptbxl_cnn_lstm_image_reproduction_seed42
```
