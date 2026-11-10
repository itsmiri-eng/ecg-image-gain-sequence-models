# Interpretation of the executed spatial CNN–LSTM

The run completed 30 epochs on the original frozen image split and selected epoch 27 using validation macro F1, balanced accuracy and weighted loss, in that priority order. Test accuracy is 71.43%, balanced accuracy 63.15%, macro F1 0.6020, weighted F1 0.7168 and macro one-vs-rest ROC-AUC 0.8944. These values were reconciled with saved predictions.

Compared with the original CNN, the model correctly recognises more test MI images (3/6 versus 1/6) and History_MI images (6/17 versus 4/17), but fewer Normal (26/28 versus 28/28) and Abnormal images (35/47 versus 41/47). LDA retains higher History_MI recall (10/17). The observed increase in macro F1 and balanced accuracy is accompanied by lower overall accuracy; it is not a general superiority result.

The LSTM receives 16 left-to-right spatial feature positions from a 2D CNN. It is not a temporal waveform model and does not perform ECG digitization. In addition to architecture, the checkpoint-selection criterion and gradient clipping differ from the earlier CNN. Their individual contributions are not isolated.

At the selected epoch, training/validation macro F1 is 0.6177/0.6001. Subsequent training improvements coincide with declining validation macro F1, but one small run does not settle the underlying causes of error. Supplied labels, patient-proxy IDs, the known crop defect and the very small MI test count remain limitations. Test results were previously observed during the baseline phase; this experiment evaluated once after selection and is exploratory. No PTB-XL data were used.

See `docs/cnn_lstm_report.md` for reproducibility, verification and the separate PTB-XL assessment.
