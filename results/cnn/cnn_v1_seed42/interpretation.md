# CNN baseline interpretation

The 40,212-parameter 2D CNN completed all 30 prespecified epochs on CPU. It used the same 295/98/98 image split as LDA, training-only global normalization, inverse-frequency class weights, AdamW, dropout, and validation-loss checkpointing. Epoch 25 had the lowest weighted validation loss and was restored for final evaluation. The seven-epoch early-stopping rule did not trigger before the 30-epoch cap; the saved model is not the last epoch.

On 98 held-out images, the CNN classified 74 correctly: accuracy 75.51%, balanced accuracy 56.86%, macro F1 0.5621, weighted F1 0.7310, and macro one-vs-rest ROC-AUC 0.9008. It classified 28/28 Normal and 41/47 Abnormal correctly, but only 1/6 MI and 4/17 History_MI. Normal precision is 0.5957 because many minority-class images are predicted Normal. Class-weighted training did not eliminate this failure.

Relative to LDA, accuracy increased by 13.27 percentage points and weighted F1 by 0.0887; balanced accuracy fell by 3.46 points and macro F1 by 0.0016. Normal/Abnormal F1 improved, while MI/History_MI F1 worsened. The CNN therefore has no demonstrated overall advantage when each class matters equally. Its higher AUC indicates better aggregate probability ranking on this split, not a successful minority-class decision rule or a validated clinical threshold.

At the selected epoch, training/validation accuracy are 72.20%/71.43%, with macro F1 0.4360/0.4607. Later epochs show some overfitting: weighted training loss fell from 0.9540 at epoch 25 to 0.8865 at epoch 30 while validation loss rose from 1.0397 to 1.0843. Validation checkpointing limited reliance on that later fitting. One split and tiny minority supports do not establish the size or stability of generalization differences.

Training took 43.70 seconds and total recorded runtime 46.47 seconds, excluding process startup and data preparation. Seed: 42. MPS was unavailable to this runtime, so the documented fallback selected CPU. The original checkpoint, architecture, history, training curve and normalization are under `models/cnn/cnn_v1_seed42/`; exact evaluation evidence is in this result directory. No test-based retuning was performed.
