# Existing project audit

Audit date: 2026-09-23. This report distinguishes material found on disk from claims made in earlier manuscripts. All original files were preserved. Newly executed baselines and the current dataset audit are documented separately; historical metrics below are not new experimental results.

## What already existed

The original folder was a collection of datasets, manuscripts and presentation material rather than a runnable research repository. A recursive inventory including hidden files found two Python files and no Jupyter notebooks: the project script `.ipynb_checkpoints/weirdwithseed-checkpoint.py` and the PTB-XL distribution's `example_physionet.py`. The latter is a dataset loading example, not a trained classifier. No original saved model weights, training history, prediction CSV, frozen image split, dependency file, README or experiment registry were found locally. The two manuscript ZIPs contain LaTeX, bibliography and figures, but no executable training code or checkpoints.

The original goal in `proj desc.pdf` is classification directly from paper ECG images. The hidden Python script instead reconstructs approximate one-dimensional signals from images before classification. The presentation calls the images a private Swiss clinical dataset, whereas the later manuscripts call them Mendeley images. Source provenance must be based on the dataset audit, not the presentation's unsupported description. The local PTB-XL directory holds waveform data; it is a distinct input modality and cannot silently replace the four-class image cohort.

## Existing image preprocessing and CNN

The hidden script specifies classes `Normal`, `MI`, `Abnormal`, and `History of MI`. The on-disk fourth folder is spelled `HIstory of MI`. This mismatch is a portability defect that can omit the class on a case-sensitive filesystem; on a case-insensitive macOS volume it may still resolve. It also enumerates files using unsorted `os.listdir`, so seed 42 alone does not freeze the sample-to-split assignment across environments.

Preprocessing implemented in that script:

1. Read grayscale and resize the entire page to 1065 by 560 pixels.
2. Compute a quality proxy from the image's central horizontal row, restricted to the leftmost quarter. A length-three median filter defines a smoothed intensity sequence; the logarithmic ratio of smoothed intensity energy to residual energy is called SNR. Images below 5 dB are filtered.
3. Estimate amplitude scale using a vertical projection of the left image quarter and peak spacing, then search the leftmost 8% of the page for a contour taller than 40 pixels and narrower than 50 pixels. A nominal 5.6 pixels/mm fallback is used when grid detection fails.
4. Apply inverted Otsu thresholding and skeletonization to the whole page, with no validated text or grid removal.
5. Divide the page into three equal row bands and four equal columns. For every column within each tile, take the average row coordinate of all skeleton pixels. Missing columns use the tile's midpoint. Mean-center, apply the heuristic scale and interpolate to 1,000 samples, yielding a 12 by 1,000 tensor.

These are approximate image-derived traces, not validated ECG waveform reconstructions. Grid lines, printed text and multiple traces can affect the column average. Fixed page tiling does not establish lead identity, time calibration or waveform fidelity. There is no comparison with paired ground-truth waveforms. The manuscript's later wording correctly says the intensity proxy is not clinical signal SNR; the earlier script and presentation overstate its interpretation.

The implemented `ECGNet` uses a 12-to-64 initial Conv1d (kernel 7, stride 2), batch normalization and ReLU, followed by three residual blocks with output widths 64, 128 and 256 (kernel 5; strides 1, 2 and 2), global average pooling and a four-output linear layer. Residual projections use 1 by 1 convolutions where required. Training uses Adam at 0.0005, unweighted cross-entropy, batch size 16 and ten fixed epochs. Python, NumPy and PyTorch seeds are set to 42. Device selection supports CUDA or CPU only, not Apple MPS.

The split is a stratified 80/20 image-level train/test split with no validation set, patient grouping or duplicate handling. There is no early stopping, learning-rate schedule, augmentation, checkpoint save, epoch-history save, dependency capture or central result log. Extraction runs again on every dataset access. Evaluation prints MCC and accuracy in the greater-than-15/25-dB subsets, then displays a confusion matrix. `classification_report` is imported but never called; the code does not save full per-class metrics or predictions. Empty quality subsets are not explicitly guarded.

## Prior numerical claims and evidence limits

| Source | Historical claim | Evidence available locally |
|---|---|---|
| `report april 9 .pdf` | 928 images; 742/186 train/test; 86.02% accuracy and MCC 0.8268; quality subset 87.13% | Manuscript text and figures; hidden script broadly matches the stated ten-epoch architecture, but no run artifacts establish provenance |
| `ecp google doc.pdf` | Enhanced 94.62% accuracy/MCC 0.9287; external fine-tuning 31.43% | Draft mixes old and enhanced results and contradictory class recalls; do not use as a verified result table |
| `ecg_manuscript_corrected_v2.zip` | 94.62% within-source result; PTB-XL fine-tuning uses a 280-image test partition | LaTeX and figures; executable experiment and split unavailable locally |
| `ecg_manuscript_corrected_v3.zip` and `ECG_Paper___Miriam.pdf` | 94.62% accuracy/MCC 0.9287 on 186 images; PTB-XL best full-cohort zero-shot 25.50%; fold-10 fine-tuning 20.00% | Later manuscript, tables and figures; no local trained model, source predictions or frozen splits for independent verification |

The later manuscript reports source counts Normal 284, MI 239, Abnormal 233 and History of MI 172 (total 928), with training counts 227/191/186/138 and test counts 57/48/47/34. It reports recall 53/57 for Normal, 48/48 for MI, 41/47 for Abnormal and 34/34 for History of MI. These are historical cohort claims, not the deduplicated cohort used for the new baselines.

The v3 manuscript's model is materially different from the available Python: four residual blocks through 512 channels, block dropout 0.1 and output dropout 0.3, 30 epochs, Adam learning rate 0.001 with weight decay 0.0001, cosine annealing, class-weighted loss, Gaussian noise and amplitude augmentation. The associated implementation is not present locally. It would therefore be incorrect to present the hidden script as exact reproduction code for 94.62%.

The v3 LaTeX explicitly leaves a TODO to confirm patient-level versus record-level splitting for Mendeley. It says filenames lack explicit patient identifiers and warns that related records require grouping. No original image deduplication code or duplicate report was found. Consequently the original 80/20 metrics cannot establish independent-patient performance, and they should not be directly compared with the new split as if only the model changed.

For PTB-XL, v3 claims one record per patient, 600 rendered images (150/class), folds 1–8 training (502), fold 9 validation (53), fold 10 testing (45), and zero pairwise patient overlap. Its mapping uses diagnostic MI statements and infarction stages for active/history labels. These are statements in a manuscript, not verified run records. The earlier v2 instead describes a different 280-image fine-tuning test partition, illustrating why versioned sample manifests are essential.

## Duplicated and superseded material

Both manuscript ZIPs contain the same set of fourteen PNG figure filenames plus a PDF figure, with revised LaTeX text. They are historical revisions to preserve, not independent experiments. The April report, Google-doc export, corrected ZIPs and September PDF contain overlapping narrative and differing results; their dates and version names are not proof that an experiment ran. Two loose architecture PNGs provide illustrations, not checkpoints or metric evidence. The image ZIP and expanded image folders are redundant storage of source images; the dataset audit determines their exact relationship without duplicating them again.

The PowerPoint is an older ten-slide overview of the hidden-script pipeline. Its performance slide lists the project's result as TBD and its final slide asks to run the code. Its claims of private Swiss data, calibrated millivolts and clinical-quality gating should not be carried forward as established facts.

`ECG_Simple_To_Do.xlsx` contains an older publication-first plan, including PTB-XL-only training, clinician review of class labels and repeated seeds. It is a task tracker, not an experiment log. The newer `ECG_Project_Logbook.docx` puts dataset validation and one-model-at-a-time reproducibility first, matching the current user instructions.

## Changes justified by the audit

The new baselines need a single audited manifest and saved train/validation/test split, exact duplicate handling and available record/patient grouping before model comparison. A compact direct-image CNN and training-only PCA followed by LDA are defensible new baselines for the stated paper-image objective. They should be labeled as new experiments rather than a reproduction of the manuscript's 1D ResNet result. The original extraction and architecture remain available for a later controlled digitization comparison after their validity can be assessed.

Existing files were not rewritten or deleted. New organization, metrics and execution outcomes are described by the project README, dataset audit and results directories. LSM and the supervisor's uncertain convolution term remain explicitly unresolved: the logbook contains placeholders and no reviewed local source defines them.

## Access limitation

The September PDF cites `https://github.com/itsmiri-eng/ecg-image-domain-shift`. Attempts to retrieve the repository through the web tool returned cache/fetch failures, and the sandbox network command could not resolve the API host. A requested network approval was interrupted. The repository's current contents, history and remote experiment artifacts were therefore not inspected or verified. No assertion about their absence remotely is made.
