# Historical material and unresolved method audit

Audit date: 1 November 2026. This is a new, read-only reconstruction of historical project evidence. No source document, dataset, split, checkpoint, prediction, configuration, or experiment log was changed, and no training or inference was run for this audit. Existing September audit reports were used as navigation aids and checked against original material. This report complements the separate audit of the current runnable pipeline and its five formal experiments.

## Findings that affect the next research decision

1. **LSM remains undefined.** The original logbook explicitly leaves it as a placeholder. No reviewed local source provides its expansion, architecture, or paper. It must not be silently expanded to a familiar abbreviation.
2. **The remembered convolution method remains undefined.** The later model-progression note records the sound as “Portal convolutions,” explicitly unconfirmed. The logbook says only “Convolution approach.” The project contains several named convolution methods, but none is tied to that supervisor instruction.
3. **There is a useful literature lead, not an identification:** the methodology review discusses *adaptive rotated convolution (ARC)* and cites Wang and Yang (2025), whose paper combines rotated lead detection, digitization, and an S4-augmented transformer. The local documents do not say this was the intended method.
4. **Historical manuscript results are not the current formal run results.** The original script is a ten-epoch, three-block 1D ResNet over heuristic image-derived traces. The later 94.62% manuscript describes a materially different four-block, thirty-epoch model. Its corresponding implementation and original run artifacts were not found locally.
5. **Publication versions are not interchangeable.** The v3 LaTeX archive retains stale figures from v2, including a 280-image E7 matrix despite v3 text describing a 45-image E7 test. The later September PDF contains replacement figures and further text changes. A source-of-truth publication package still needs to be reconciled before reuse.

## Coverage and reading method

The seven root PDFs were extracted page by page with the bundled `pypdf` runtime, for 97 pages in total. Relevant methods, results, limitations, references, and numerical conflicts were inspected. Key figure conflicts were checked visually from the original image entries or rendered PDF pages. This was a scientific-content audit, not a page-by-page layout certification. One extraction emitted a rotated-text warning; graphics and rotated text were not treated as exhaustively searchable.

DOCX body/table paragraphs, PowerPoint slide and notes XML, and worksheet cell values were read directly from their OOXML containers. Both manuscript ZIP inventories and every contained LaTeX/bibliography source were inspected, including comments omitted from PDFs. ZIP figure bytes were compared by SHA-256 without altering the archives. The image ZIP was inventoried without expanding the dataset again. Repository filenames, source comments, notes, configs, logs, and extracted historical text were searched for `LSM`, `portal`, `liquid`, `state space`, `rotated convolution`, convolution terms, placeholders, and TODOs. Generated audit files from this turn were excluded from the method search.

| Material | Coverage and role |
|---|---|
| `proj desc.pdf` | 2 pages. Original objective: classify paper/scanned ECG images directly, explore CNNs/transformers, augmentations, transfer, benchmark and real-scan evaluation. Its statements about restricted paper PTB-XL scans are proposal text, not evidence that such data were acquired locally. |
| `.ipynb_checkpoints/weirdwithseed-checkpoint.py` | All 245 lines, including comments. Only recovered original project-specific executable script. No embedded outputs, saved checkpoints, or result history. |
| `notebooks/README.md` | Notebook policy inspected. No `.ipynb` notebooks found in the repository filename inventory. The checkpoint-named Python file is a notebook export, not a notebook with cell outputs. |
| `ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/example_physionet.py` | Entire distribution example inspected: loads waveforms and metadata, aggregates diagnostic superclasses, uses folds 1–9 versus 10. No classifier or trained result. |
| `ECG_Project_Logbook.docx` | All 40 paragraphs, including empty ongoing-log table. Explicit placeholders and the one-model-at-a-time reproducibility requirements. |
| `ECG_Simple_To_Do.xlsx` | `To Do` sheet, used cells A1:E18, including status and completion criteria. No formula-backed experiment results. |
| `ECG_Project_Presentation.pptx` | All 10 slides and 10 notes entries. Notes contain slide numbers, no additional method clarification. Older presentation of hidden-script pipeline; slide 8 says project score TBD; slide 10 asks to run it. |
| `report april 9 .pdf` | 14 pages. Earlier 86.02% result, three-block/ten-epoch method, calibration narrative, ablations, quality strata. |
| `ecp google doc.pdf` | 19 pages. Intermediate 94.62% source result and older 31.43% fine-tuning claim; mixed old/new methods and metrics. |
| `ecg_manuscript_corrected_v2.zip` | All 19 entry names; `ecg_manuscript/main.tex`, `references.bib`, 14 PNGs and one PDF figure. No training code or run data. |
| `ecg_manuscript_corrected_v3.zip` | All 19 entry names; `manuscript_v2/ecg_manuscript/main.tex` and bibliography inspected, relevant figures viewed. No training code or run data. |
| `ECG_Paper___Miriam.pdf` | 17 pages. Later source/domain-shift manuscript, replacement fold-10 plots, GitHub data-availability statement. Not an exact rendering of retained v3 archive. |
| `AI_Transforms_Paper_ECGs___Methodology_paper.pdf` | 24 pages. Review manuscript reporting 50 eligible studies; searched methods/tables/references and inspected convolution lead. Not a local experiment record. |
| `ECG_from_paper_to_diagnosis.pdf` | 17 pages. Actual title is *ECG Sonification Methods for Robust and Generalizable Clinical Decision Support*. Peripheral literature review, not the image-classifier paper. |
| `physionet paper.pdf` | 4 pages. Primary Krones et al. SignalSavants paper on Hough rotation, segmentation, and waveform reconstruction; page 1 rendered to verify DOI. Third-party result. |
| `architecture.png`, `architecture_overview.png` | Both inspected visually. Image-to-trace, calibration, skeletonization, 1D ResNet and domain-shift illustrations; no LSM definition. They are diagrams, not validation evidence. |
| `mendely images.zip` | 932 entries: 928 JPEGs and four directories; no code or notes. Dataset provenance and image-level audit are handled separately. |
| `docs/`, `README.md`, `src/`, `tests/`, `experiments/` | Current textual material searched for unresolved names, plans and references. Key progression, handoff, research-direction and prior source-audit documents read. Current-run technical reconstruction is outside this report's bounded historical role. |

The local material contains a remote repository URL, but this audit does not infer its current contents or remote experiment provenance. Any separately completed remote inspection should be reported on its own evidence.

## Exact unresolved-name evidence

| Source locator | Evidence | Consequence |
|---|---|---|
| Logbook, Word paragraph 12 | “LSM — placeholder; confirm the full name and method before implementation.” | No definition or expansion authorized by this source. |
| Logbook, Word paragraph 13 | “Convolution approach — placeholder; confirm the intended method first.” | No specific convolution variant identified. |
| `docs/model_progression.md:4–5` | LSM is unconfirmed; “Portal convolutions” is the remembered sound, not an identified method. | A phonetic recollection is not a method specification. |
| `experiments/configs/next_model_protocol.json:2–9` | Status `awaiting_confirmed_method`; confirmed name and definition/reference are null; note explicitly prohibits expansion/interpretation without confirmation. | This preserved preparation file is not an executable method definition. |
| `docs/project_status.md`, `docs/research_direction.md`, `docs/next_model_handoff.md` | Continue to record both names as unresolved. | No later local clarification supersedes the logbook. |
| Methodology review PDF p.7 and ref.43 on p.21 | ARC is named; ref.43 is Wang and Yang, DOI `10.3390/sym17010120`. | Contextual lead only; no link to supervisor intent. |
| Presentation slides 9–10 | Recommends ConvNeXt-Tiny/EfficientNet-B4, a direct image baseline, and Hough rotation correction. | These are explicit presentation recommendations, but none is labelled as the unresolved progression step. |

The publisher independently confirms the Wang–Yang method uses adaptive rotated convolutions for lead detection and an S4-augmented transformer after digitization. That does not define LSM or establish that ARC was the remembered method. [Publisher article](https://www.mdpi.com/2073-8994/17/1/120).

## Original implementation reconstructed

The hidden Python script resizes the full grayscale page to 1065×560, estimates an image-intensity quality ratio from the central row's left quarter with a median filter, and filters at 5 dB. It estimates scale from grid-peak spacing and the first narrow/tall contour in the leftmost 8% of the page, with a 5.6 pixels/mm fallback. Otsu thresholding and skeletonization act on the full page. A fixed 3×4 partition turns each tile into a column-wise mean skeleton position, fills missing columns at the midpoint, mean-centres/scales the result and resizes it to 1,000 positions. The result is 12×1,000 approximate image-derived traces. There is no paired-waveform validation, established lead identity, or validated time axis.

`ECGNet` has a 12→64 Conv1d stem (kernel 7, stride 2), three residual blocks at 64/128/256 channels (kernel 5), global average pooling and four outputs. It uses Adam at 0.0005, unweighted cross-entropy, batch size 16, ten fixed epochs and seed 42. The image-level 80/20 stratified split has no validation partition, patient grouping, or duplicate safeguards. Unsorted `os.listdir` means the seed alone does not freeze the same sample assignment across environments. The class path uses `History of MI` while the actual folder is `HIstory of MI`, a case-sensitive portability issue.

It prints threshold-stratified accuracy and MCC and displays a confusion matrix. It does not persist model weights, predictions, a manifest, learning curves, fitted metadata, or environment provenance. `classification_report` is imported but never called. No run was executed during this historical audit.

The v3 Methods instead specify four residual blocks through 512 channels, dropout, thirty epochs, learning rate 0.001, weight decay, cosine annealing, class weights, Gaussian noise and amplitude augmentation. This exact enhanced implementation is not the recovered script. The filenames `ptbxl_generate.py` and `ptbxl_analysis_b_metrics.json`, referenced by v3, were not found anywhere in the current local file inventory.

## Historical claims requiring original run evidence

These values are recorded to reconstruct the narrative. They must remain labelled historical manuscript claims until linked to their own source code, exact manifests/splits, checkpoints and predictions. The five newer formal runs cannot retrospectively validate them.

| Claim family | Historical evidence | Evidence still needed |
|---|---|---|
| Early within-source result | April report: 928 images; 742/186 train/test; 86.02% accuracy, MCC 0.8268; higher-quality subset N=171, 87.13%, MCC 0.8385. | Original per-record split, predictions, checkpoint, training records; duplicate/group audit for that exact split. |
| Enhanced within-source result | Intermediate/later drafts: 94.62% accuracy, MCC 0.9287, N=186; later table reports Normal 53/57, MI 48/48, Abnormal 41/47, History 34/34. Higher-quality subset 94.15%, MCC 0.9217, N=171. | Enhanced four-block implementation and original fitted artifacts; patient/group independence; prospective versus post hoc quality-threshold decision. |
| Pipeline-component comparison | Later draft: argmin/fixed scale 83.33%/0.7790; Canny/skeleton 92.47%/0.8995; contour/grid/skeleton 94.62%/0.9287. Earlier draft has different Canny/final results. | Per-variant code and equal-data/equal-training controls. Multiple components changed, so these cannot isolate calibration's causal contribution. |
| Rotation/blur/brightness robustness | v3 Table 4 gives +5° 30.65%, +15° 22.58%, +30° 26.34%; blur k=3/7/15 gives 52.15/26.88/27.96%; brightness −40% retains 94.62%. | Saved transformations, exact original predictions and inference outputs; no such formal historical run package found. |
| Historical PTB-XL zero-shot | v3: 600 rendered images, 150/class; E1–E6 best full-cohort 25.50%, best fold-10 28.89%. | Original rendered cohort, generator, class mapping, selected patient IDs, original splits and all variant predictions. |
| Historical PTB-XL fine-tuning | Google-doc export: 31.43%/0.0858. v2: 31.07%/0.0812 on N=280. v3 and later PDF: train/validation/test 502/53/45 and E7 20.00%/−0.051. | Version-specific run artifacts. These are distinct claimed cohorts/results, not one interchangeable result. |
| Physical/clinical fidelity | Early report, slides and diagrams claim calibrated millivolts, clinical-quality filtering, robust/valid reconstruction. | Paired waveform/measurement validation and readability adjudication. The later paper correctly describes Q as an unvalidated image-quality proxy. |
| Dataset provenance | Slides call the cohort private Swiss scans; manuscripts call it Mendeley. | Use the actual archive/data provenance audit. Slide assertions cannot establish origin or privacy status. |

## Concrete publication defects and version differences

- **v3 archive stale E7 figure:** `figures/cm_ptbxl_best.png` is byte-identical in v2 and v3. It visibly reports 31.07%, MCC 0.0812 and matrix rows `[17,14,16,23]`, `[12,24,19,15]`, `[15,22,21,12]`, `[13,23,9,25]`. Each row sums to 70, so N=280 and the diagonal is 87. In contrast, v3 `main.tex:982–988` describes 20.00%, MCC −0.051, N=45 and explicitly claims the matrix sums to 45.
- **Other archive figures are also unchanged:** all 14 PNGs, the PDF figure and bibliography are byte-identical between v2 and v3; only `main.tex` differs. `domain_shift_summary.png` still plots E7 near 31% with positive MCC, inconsistent with the revised v3 E7 table. This is evidence of stale figures, not evidence that the revised experiment ran.
- **v3 Figure 1 caption mismatch:** `main.tex:720–731` describes two high/low quality reconstructed-signal examples with Q=34.2 and Q=20.0, while `figure1_pipeline.png` visibly contains three panels labelled Original Scan, Binarized Mask and Skeletonized Extraction. The source itself retains a TODO to verify what the figure shows.
- **September PDF must be handled separately:** p.14 contains replacement fold-10 E2/E7 plots, with E7 title 20.00% and −0.051. Thus the stale N=280 image finding is specific to the v3 ZIP. The September PDF p.6 supplies the GitHub URL, while v3 source says code is available on request and retains a repository TODO.
- **September PDF E2 caption overstates collapse:** p.14 Figure 7 says all E2 predictions collapse to Abnormal, but the visible Normal row contains one MI prediction and ten Abnormal predictions. Correct wording would need the complete saved prediction record; do not infer one-class prediction from the narrative.
- **Intermediate export mixes revisions:** the Google-doc PDF abstract says perfect Normal/MI sensitivity, while its later table has Normal recall 0.930; p.7 calls a decrease from 94.62% to 94.15% an improvement. Its ablation table mixes older recalls with newer accuracy. This export is unsuitable as a definitive result table.
- **Krones bibliography DOI conflict:** both ZIP bibliographies give `10.22489/CinC.2024.072`. The primary local paper's first-page footer clearly prints **10.22489/CinC.2024.097**. The primary source should control the correction.
- **ECG-Image-Kit reference conflict:** methodology review ref.30 attributes the toolbox to Golany/Radinsky (2020), arXiv:2008.03660. The manuscript bibliography and authoritative PubMed record instead identify Shivashankara et al. (2024), DOI **10.1088/1361-6579/ad4954**. [PubMed record](https://pubmed.ncbi.nlm.nih.gov/39150768/).
- **Bibliography header is stale:** it claims 35 verified references, but the retained bibliography contains 41 entries. A comment claiming verification is not proof that all entries are correct.

These are audit findings only. No manuscript or old figure was repaired during this work.

## Unfinished work found in the historical material

The v3 LaTeX contains 20 lines mentioning TODO, including one prose pointer to “Methods TODO.” Its open items include:

- Confirm author affiliations, correspondence email, equal-contribution/ORCID notes, funding attribution, competing interests and the target journal's ethics statement requirements. The text already asserts author approval and no conflict; those assertions are not evidence that its remaining confirmations happened.
- Establish whether the original Mendeley split was record-level or patient-level; recompute if related images crossed partitions.
- State whether the 5/25 dB quality thresholds were chosen prospectively or after inspecting the data.
- Record exact software versions and hardware; supply learning curves and confidence intervals for headline metrics.
- Supply the implementation, preprocessing parameters, original label mapping, weights where permitted, exact split files, environment and reproduction instructions.
- Verify Figure 1's content and caption; reconcile all figures against the specific experiment version.
- The limitations also call for repeated seeds, a calibration-only ablation, independent real scan/photo validation, and evidence distinguishing rendering/population/label-mapping shifts. These are proposed studies, not completed local results.

The spreadsheet preserves an older publication-first plan: supervisor review (row 5, “Doing”); author/venue decisions and submission (rows 6–7, “Not started”); PTB-XL-only training, error analysis, clinician review, multiple seeds/folds, combined-source training, realistic scan/phone augmentation, matched waveform comparison, hospital partnership and paired-data collection (rows 8–16, “Not started”). Those statuses are historical and should not override later completed runs. Several later formal experiments partially address the technical rows, while clinician sign-off, repeated-seed stability, independent validation and submission are not established by this tracker.

The later logbook prioritizes dataset validation, grouped saved splits, train-only fitting, one model at a time, full artifact logging and model comparison. Its ongoing-results table is blank. It is a plan, not the current experiment registry. The September handoff describes a clipped-waveform case, an unadopted one-case crop preview and an unfilled clinical-review queue; it does not establish cohort-wide replacement preprocessing or clinical adjudication. Current research-direction notes explicitly retain repeated-seed and independent-data assessment as future work.

## Reusable primary literature already referenced in the project

These references are useful starting points for a separately scoped literature review. Inclusion here is not a model recommendation, verification of every published result, or evidence that a local run reproduced the paper. The publisher lookup for ARC and PubMed lookup for ECG-Image-Kit were checked on the audit date; other entries below are transcribed from retained sources unless stated otherwise.

| Primary reference | Local locator | What it can support |
|---|---|---|
| Wang & Yang (2025), *A Systematic Method Combining Rotated Convolution and State Space Augmented Transformer for Digitizing and Classifying Paper ECGs*, [publisher](https://www.mdpi.com/2073-8994/17/1/120) | Methodology review p.7/ref.43 | ARC lead detection and S4-based post-digitization classification. Context for the unresolved convolution lead only. |
| Krones, Walker, Lyons & Mahdi (2024), *Combining Hough Transform and Deep Learning Approaches to Reconstruct ECG Signals From Printouts*, DOI 10.22489/CinC.2024.097 | Full primary paper retained as `physionet paper.pdf` | Hough rotation, segmentation and waveform reconstruction; digitization validation, distinct from direct diagnosis. |
| Shivashankara et al. (2024), *ECG-Image-Kit*, [authoritative record](https://pubmed.ncbi.nlm.nih.gov/39150768/) | Both manuscript bibliographies; correction to review ref.30 | Synthetic ECG-image rendering and artifacts from waveform inputs. |
| Wagner et al. (2020), *PTB-XL, a large publicly available electrocardiography dataset*, DOI 10.1038/s41597-020-0495-6 | Both bibliographies, later PDF ref.14, bundled distribution | Dataset design and provenance; use installed release metadata for local counts. |
| Strodthoff et al. (2021), *Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL*, DOI 10.1109/JBHI.2020.3022989 | Both bibliographies; later PDF ref.15 | Waveform benchmarks and evaluation conventions, not a same-modality image baseline. |
| Dias et al. (2024), *Image-Based Electrocardiogram Classification Using Pre-trained ConvNext* | Methodology review ref.31; slides 8–9 | A direct-image convolution literature candidate requiring primary-paper verification before relying on reported scores. |
| Stenhede et al. (2024), *Digitizing paper ECGs at scale: an open-source algorithm for clinical research*, DOI 10.1038/s41598-024-79901-0 | Both bibliographies | Digitization method and clinical-research extraction context. |
| Baydoun et al. (2019), *High Precision Digitization of Paper-Based ECG Records*, DOI 10.1109/JTEHM.2019.2949784 | Both bibliographies; Krones reference list | Recovery/measurement fidelity, not evidence that the local heuristic is physically validated. |
| He et al. (2016), *Deep Residual Learning for Image Recognition*, DOI 10.1109/CVPR.2016.90 | Both bibliographies | Residual architecture foundation; generic inspiration rather than exact reproduction of the hidden 1D network. |
| Zhang & Suen (1984), DOI 10.1145/357994.358023; Otsu (1979), DOI 10.1109/TSMC.1979.4310076 | Both bibliographies | Thinning and thresholding components. They do not validate lead segmentation, calibration or extracted physiology. |

No evidence in this material resolves LSM or the remembered convolution name. Their definition remains a specific missing piece of supervisor input; unrelated completed CNN-LSTM/spatiotemporal work does not fill that gap.
