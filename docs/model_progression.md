# Supervisor's model progression

**Update, 1 November 2026 evening.** The progression now runs on the main PTB-XL image benchmark as well, with identical inputs, folds and evaluation:

| Step | PTB-XL run | Status | Test macro AUROC |
|---|---|---|---:|
| 1 LDA | `ptbxl_v2_lda_seed42` (PCA 256) and `ptbxl_v2_lda_wide_seed42` (PCA 512, validation optimum) | Complete | 0.7883 / 0.7939 |
| 2 LSM | — | Unresolved; not implemented | — |
| 3 Convolution method | — | Unresolved; possibly "temporal convolution" (already Run 5) — ask supervisor | — |
| 4 CNN | `ptbxl_v2_cnn_pool_seed42/43/44` | Running | pending |
| 5 CNN + LSTM | `ptbxl_cnn_lstm_image_v1_seed42` (+ seeds 43/44 running) | Complete (seed 42) | 0.8698 |
| 6 Small spatiotemporal network | `ptbxl_spatiotemporal_image_v1_seed42` (+ seeds 43/44 running) | Complete (seed 42) | 0.8803 |
| 7 ViT (compact hybrid) | `ptbxl_hybrid_vit_image_v1_seed42` (+ seeds 43/44 running) | Complete (seed 42), verified | 0.8902 |

See [master comparison](../results/comparisons/ptbxl_master/master_table.md) and [status](PROJECT_STATUS.md). The historical text below is kept unchanged.


1. **LDA** — first current baseline: fixed cropped ECG pixels, train-only PCA, shrinkage Linear Discriminant Analysis.
2. **LSM — UNCONFIRMED.** Obtain the exact full name and reference from the supervisor. No implementation or expansion is assumed.
3. **“Portal convolutions” — UNCONFIRMED.** This is the remembered sound of the term, not an identified method. Obtain the exact name/reference before implementation.
4. **CNN** — second current baseline: small 2D image CNN, validation-based early stopping.
5. **CNN + LSTM — two separate image experiments completed on 24 September 2026.** Mendeley run `cnn_lstm_image_v1_seed42` achieved four-class test accuracy 71.43%, macro F1 0.6020 and balanced accuracy 63.15%; see its [report](cnn_lstm_report.md). PTB-XL run `ptbxl_cnn_lstm_image_v1_seed42` achieved five-label test macro AUROC 0.8698, macro average precision 0.7035 and macro F1 0.6131 at threshold 0.5; see the [PTB-XL report](ptbxl_cnn_lstm_report.md). These scores concern different cohorts and tasks. Both models process image-feature sequences; neither substitutes native waveform samples for images. A waveform-input variant remains a separate future experiment.
6. **Small spatiotemporal network — completed and verified, 26 September 2026 locally (25 September, 23:17 UTC).** The same PTB-XL spatial CNN feeds a compact temporal-convolution head over 70 aligned image-feature positions. The 17,161-parameter model ran 20 epochs and restored validation-selected epoch 16. Test macro AUROC is **0.8803 versus 0.8698** for CNN–LSTM; the paired gain is **0.0105** (95% patient-bootstrap interval **0.0045–0.0167**). Macro F1 is **0.6409 versus 0.6131**. See the [frozen protocol](</Users/miriamistanboli/Downloads/ecg swiss/docs/ptbxl_spatiotemporal_protocol.md>), [completed report](</Users/miriamistanboli/Downloads/ecg swiss/docs/ptbxl_spatiotemporal_report.md>) and [paired comparison](</Users/miriamistanboli/Downloads/ecg swiss/results/comparisons/ptbxl_spatiotemporal_image_v1_seed42/comparison.md>).
7. **Compact hybrid image transformer — training, 1 November 2026.** After auditing the five prior runs, a 23,881-parameter CNN-feature transformer was prespecified for the existing 17,084 training ECGs. It retains the prior CNN and compares global self-attention over 70 feature positions under the same training budget. A synthetic resource probe and model/runner tests passed before fitting. This is a small hybrid experiment, not a large pretrained ViT. See [frozen protocol](ptbxl_hybrid_vit_protocol.md) and [current status](PROJECT_STATUS.md).

LSM and the remembered convolution term remain unresolved. **Five model runs are complete**: three Mendeley baselines and two PTB-XL image models. A sixth hybrid ViT run is now active; its test outcome is pending. The [PTB-XL dataset](ptbxl_image_dataset.md) contains 21,799 full-duration twelve-lead pages, with 21,388 eligible for five-target classification under official patient folds. The fifth run records one final test inference after selection; saved-artifact verification and all **152 tests pass**. Its **164 protected prior files and four prior log rows are preserved**.

The spatiotemporal model is the **provisional stronger overall PTB-XL candidate**, with validation macro AUROC also higher (0.8889 versus 0.8816), but it is not a universal best model. Test MI recall improves from **43.27% to 56.18%**, while HYP recall declines from **32.44% to 27.86%**, with **189 HYP positives missed**. HYP recall declines on validation too. The same already-observed test cohort makes this comparison exploratory, and changing head capacity and dropout placement prevents a pure architecture-mechanism conclusion.

After the current hybrid comparison, the next work is matched PTB-XL CNN/LDA baselines and a prespecified repeated-seed study using development data, followed by independent-data assessment of a frozen approach. Neither has been performed. Keep the representation, targets, patient partitions and evaluation rules explicit; any threshold or preprocessing changes require their own development protocol. A new full run requires an updated protocol and prior-log preservation baseline, because the completed spatiotemporal protocol locks four prior rows and the current log has five. Clean rendered-page performance does not establish performance on real scans. See the [research direction](</Users/miriamistanboli/Downloads/ecg swiss/docs/research_direction.md>).

## Historical readiness update — before CNN–LSTM implementation

The post-baseline technical reviews are complete and linked from [next-model handoff](next_model_handoff.md). Duplicate/group checks pass, while clinical labels and patient-ID semantics remain unverified. Validation error analysis confirms weak CNN minority recall, and an input spot-check found crop-induced waveform loss. A one-case alternative crop preview is not a validated replacement. These limitations must accompany the next experiment.

At that stage, the next-model protocol pinned the existing dataset and split and left the model definition and training specification empty for unresolved LSM. The user subsequently requested CNN–LSTM, now executed under its separate `cnn_lstm_image_v1.json` protocol. The older placeholder is retained as historical evidence; it is not the configuration used for the completed third model.
