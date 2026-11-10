# Roadmap to a Q1-quality paper (2 Nov 2026)

Written after all planned runs finished on the PC. Goal: a rigorous, reproducible comparison of image
models for 12-lead ECG classification that a strong journal (for example IEEE JBHI, IEEE TBME,
Computers in Biology and Medicine) would accept. The manuscript itself waits until the evidence is
complete and the supervisor has seen it.

## Where the evidence stands

- Five model families on 21,388 rendered PTB-XL pages, official patient folds, three seeds for every
  network (42/43/44), one test inference per run, patient-cluster bootstrap.
- Three-seed test macro AUROC (mean ± SD): CNN-only 0.829 ± 0.005, CNN-LSTM 0.879 ± 0.008,
  temporal conv 0.879 ± 0.001, hybrid ViT 0.888 ± 0.002; PCA + LDA 0.794 (one seed).
  **The seed-42 claim "each step adds about 0.01" does not survive seeds:** CNN-LSTM and temporal
  conv are tied. The robust ordering is LDA < CNN-only < {CNN-LSTM, temporal conv} < ViT.
- Gain shortcut: the renderer's page gain leaks hypertrophy. On test pages re-drawn at one fixed
  gain, every adaptive-gain network loses 0.002–0.008 macro AUROC (LDA 0.043), and HYP sensitivity
  rises on re-drawn pages. Retraining on fixed-gain pages raised macro AUROC by 0.009–0.019 and HYP
  AUROC clearly (seed 42; seeds 43/44 now training to remove the CPU-vs-GPU confound).

## The weaknesses a reviewer will find, and the fix for each

| # | Weakness | Fix | Cost |
|---|---|---|---|
| 1 | **No untouched test set.** Fold 10 was seen in September, so every PTB-XL result is exploratory. This is the biggest problem. | External confirmatory test, analysis plan frozen before the data is opened: (a) other public 12-lead datasets (PhysioNet/CinC 2021: Georgia, Chapman-Shaoxing, Ningbo, CPSC) drawn with the same renderer, with a SNOMED → superclass mapping fixed in advance; and/or (b) real printed, scanned and photographed ECG pages (PhysioNet ECG-Image-Database; its PTB-XL coverage and licence to be checked before planning), restricted to fold-10 records, to test real-paper domain shift. | Free downloads (needs your OK); 1–2 days of code; GPU inference only |
| 2 | No reference for what the image representation costs. | Signal-domain model (1D ResNet on the raw PTB-XL waveforms, same folds, labels and protocol; `wfdb` is installed). | ~1 day; a few GPU hours |
| 3 | Three seeds give a rough SD; no correction for many comparisons. | Seeds 45/46 for every network (≈8 GPU hours); seed-level + patient-level (hierarchical) bootstrap; Holm correction across pairwise comparisons. | Mostly GPU time |
| 4 | Capacity confound: CNN-only has 1,705 parameters, the sequence heads 17–24k. | Capacity-matched CNN-only head (≈20k parameters, no sequence mixing), already in the plan. | ~3 GPU hours |
| 5 | Very small shared encoder (1,220 parameters); readers will ask whether conclusions hold at normal scale. | One standard baseline: ImageNet-pretrained ResNet-18 fine-tuned on the same pages. Reported as a reference, not part of the controlled head comparison. | ~1 day |
| 6 | Fixed 0.5 threshold makes F1/sensitivity look poor for rare labels. | Keep 0.5 as primary; add calibration (Brier, ECE, reliability plots) and validation-chosen thresholds as a secondary analysis. | Half a day, no training |
| 7 | "Do the models look at the waveform?" | Saliency (Grad-CAM / integrated gradients) on validation pages, especially HYP and the calibration pulse / gain area. | Half a day |
| 8 | Reproducibility package. | Git repository (you push), environment lockfile, Zenodo archive, model cards, data statement (PTB-XL is CC BY 4.0). | Half a day |

## The likely main contribution

The gain finding is the most novel result, refined on 3 Nov by the saliency, pulse-swap and matched
fixed-gain analyses: **a rendering choice in a synthetic ECG-image benchmark (adaptive page gain) both
leaks a small diagnostic prior and makes amplitude-based diagnosis depend on reading the calibration
pulse.** Architectures that see a whole row at once (ViT, CNN-LSTM) learn to read the pulse as a voltage
scale; local ones (CNN-only, short-range temporal conv) cannot, and gain most when gain is fixed. A
controlled re-rendering removes both effects without hurting any model. Combined with a controlled head comparison (RNN vs temporal convolution vs
attention on one shared encoder) and an external test, that is a clear, defensible paper. It does
not need a new architecture.

## The supervisor's terms ("vsm", "portal convolutions")

No local file defines them, and a literature search found no method called "portal convolution(s)".
Readings that fit the notes' order (vsm → CNN → CNN-LSTM → small spatiotemporal net → ViT for more data):

| Term | Most likely readings | What was done |
|---|---|---|
| "vsm" | **SVM** (support vector machine; the classic first baseline, letters swapped), or a **vision state-space model** (Mamba-style; recent 12-lead ECG work exists, e.g. ECG-Mamba) | PCA + linear SVM baseline added under its own frozen config (`experiments/configs/ptbxl_v2_svm.json`), labelled as an interpretation. A state-space head on the shared encoder is a candidate next model (it would complete the RNN / TCN / attention / SSM set). |
| "portal convolutions" | "temporal convolutions" (already run), "pointwise convolutions" (already inside the shared encoder), or "partial / dilated convolutions" | Nothing new implemented; the temporal-conv model and the encoder's pointwise convolutions cover the plausible readings. |

When you do talk to the supervisor, one line settles it: "Did you mean SVM or a state-space model
(Mamba), and was 'portal' temporal convolutions?"

## Recommended order

1. Fixed-gain seeds 43/44 (running now, ≈8 GPU hours) → rerun `src.compare_fixed_gain` on matched hardware.
2. Signal-domain 1D reference model (#2).
3. Seeds 45/46 and the capacity-matched head (#3, #4) on the GPU overnight.
4. Calibration, thresholds and saliency (#6, #7): no training.
5. Write and freeze the external-test analysis plan, then download and run the external test (#1).
6. Pretrained ResNet-18 reference (#5) and, if wanted, the state-space head.
7. Reproducibility package (#8), then the supervisor meeting, then the manuscript.
