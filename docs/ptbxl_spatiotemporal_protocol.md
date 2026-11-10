# PTB-XL small image spatiotemporal network: prespecified protocol

This is the next experiment in the supervisor's progression, following the completed PTB-XL image CNN–LSTM. The question is whether a compact convolutional sequence head provides a better practical alternative to the LSTM on the same rendered ECG task. It is a project-defined small network, not an asserted reproduction of an unspecified supervisor architecture.

The numerical specification is `experiments/configs/ptbxl_spatiotemporal_image_v1.json`. One configuration and seed 42 will be fitted. Model completion and results must be established by saved run artifacts; this protocol itself makes no performance claim.

## Inputs and preservation

Reuse the exact existing `ptbxl_lead_images_v1` cache, five overlapping labels NORM/MI/STTC/CD/HYP, and official patient folds: 17,084 training, 2,146 validation, and 2,158 test ECGs. The [PTB-XL release](https://physionet.org/content/ptb-xl/1.0.3/) documents the source recordings and patient folds. Preserve the 411 unmapped records' exclusion from classification.

Each input has shape 12 × 120 × 1120: twelve aligned lead-image rows, all ten seconds and calibration pulses retained, header/footer excluded, inverted grayscale and isotropic 2×2 maximum-ink pooling. Reuse the original training-only global normalization. No waveform values, patient identifiers, diagnostic metadata, gain numbers, augmentation or new filtering enter the model. Raster-detail loss and page-gain variation remain unchanged.

All four prior model runs, their source snapshots, saved report, dataset/cache manifests and experiment-log prefix are locked in `experiments/protocols/ptbxl_spatiotemporal_image_v1/prior_artifacts.json`. The central log receives one append-only record; failed attempts also receive records. Existing artifacts cannot be overwritten.

## Concrete architecture

The spatial encoder is identical in architecture to the previous CNN–LSTM: depthwise filters initially process individual lead channels; pointwise convolutions combine aligned leads. It produces 24 × 15 × 70 features, pooled to four vertical bands and rearranged as 70 positions × 96 features. All weights are trained from scratch; previous fitted weights are not transferred.

The new head projects 96 channels to 32, followed by two residual blocks. Each block has two kernel-3 one-dimensional convolutions, at dilation 1 in the first block and 2 in the second. Convolutions use left padding; each is followed by LayerNorm across the 32 channels independently at each position, ReLU and dropout. Each residual sum is followed by ReLU. Whole-sequence mean pooling, dropout 0.2 and a linear layer produce five independent logits. The implementation and synthetic checks establish exactly 17,161 trainable parameters before fitting, compared with 18,025 for CNN–LSTM.

Residual dilated temporal convolutions are motivated by [Bai, Kolter and Koltun's sequence-modeling study](https://arxiv.org/abs/1803.01271). Our small head, image encoder and normalization differ from that study; its results do not predict a win on these ECGs.

The head's local receptive field is 13 feature positions, followed by averaging over all 70 positions. These are horizontal image-feature positions, including calibration/margins, not reconstructed voltage samples. Causality applies only to the head given a feature sequence: the preceding symmetric image convolutions, spatial GroupNorm and whole-sequence pooling mean the full classifier is offline, not a streaming causal ECG model. Lead channels are mixed without asserting that neighbouring channel indices are anatomical neighbours.

## Fitting, selection and evaluation

Keep the previous training budget and rules: unweighted BCEWithLogitsLoss, AdamW learning rate 0.001 and weight decay 0.0001, batch 16, four CPU threads, no loader subprocesses, gradient clipping at 1, deterministic algorithms, and a cosine schedule to 0.00001. Train at most 20 epochs; stop after four without strict validation improvement. Select lexicographically by validation macro AUROC, macro average precision, then negative probability BCE; retain the earliest exact tie. Use a fixed independent probability threshold of 0.5 for every label.

Only synthetic inputs may be used for implementation tests and compute feasibility before freezing. Do not choose architecture variants using ECG outcomes. Freeze code, configuration, prior-artifact inventory and input hashes before actual training. Online training loss and full epoch-end validation metrics are distinct quantities and must be labelled accordingly.

After selection, evaluate the saved checkpoint on training and validation, confirm validation metrics reproduce the selected epoch, and replay 32 training examples with an independently reloaded checkpoint. Only then make one final prediction pass on the test set. Save probabilities, label decisions, full metrics, per-label counts/confusions, runtime, architecture, checkpoint and provenance.

The test set has already been observed in the CNN–LSTM experiment. This comparison is therefore exploratory; it is not a new untouched confirmatory test. No test-driven architecture or threshold tuning is allowed. A change in capacity, head structure and dropout placement accompanies the architecture change, so this experiment is a practical model comparison rather than proof that one mechanism alone caused a difference.

## Comparison and uncertainty

Primary metric: macro AUROC, giving each diagnosis equal weight. Also report macro average precision; macro/micro/weighted F1; per-label precision, recall and specificity; exact five-label match accuracy; and individual label-decision accuracy. Accuracy definitions must stay separate. No direct ranking against the different four-class Mendeley task is valid.

Reuse saved predictions for an aligned CNN–LSTM versus spatiotemporal comparison. Require identical ECG IDs, patient IDs, labels, partitions, normalization and threshold. Estimate new-minus-old differences with 500 paired patient-cluster bootstrap resamples, seed 2024: draw patients once per resample and include the same patient records, with multiplicity, for both models. Retain all six prespecified aggregate metrics and report per-label changes, including deteriorations. Individual-model intervals use the same established patient-bootstrap procedure.

A positive point difference alone does not establish superiority. Intervals describe sampling variability conditional on these two fitted models; they exclude training-seed variability, development choices and external-domain shift. Three or more seeds on development data and independent data are later steps before a stronger best-model claim. This experiment does not validate real scans, photographs or clinical use.

## Commands

```bash
cd "/Users/miriamistanboli/Downloads/ecg swiss"
/opt/anaconda3/bin/python -m src.run_ptbxl_spatiotemporal --run-id ptbxl_spatiotemporal_image_v1_seed42
/opt/anaconda3/bin/python -m src.verify_ptbxl_spatiotemporal --run-id ptbxl_spatiotemporal_image_v1_seed42
```

Operational note after completion: the commands above document the original execution. This runner is deliberately bound to the frozen four-run log snapshot and now refuses another fit in the five-run workspace. A future reproduction requires a new run ID and a separately versioned protocol/runner preservation baseline that includes all currently completed runs; do not replace the original lock. The existing completed or failed run must never be overwritten or have its test prediction repeated.
