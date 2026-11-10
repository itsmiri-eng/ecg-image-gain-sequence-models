# Source references and evidence map

Inspected 2026-09-23. Paths are relative to the project root unless stated otherwise. Original documents were read without editing; ZIPs were listed and selected text extracted to temporary audit files without expanding the large image archive. These are source descriptions, not endorsements of every claim they contain.

| Source | Role and inspected evidence |
|---|---|
| `.ipynb_checkpoints/weirdwithseed-checkpoint.py` | Only original project-specific Python implementation found, including hidden paths. Defines the image quality proxy, page-to-trace extraction, three-block 1D ResNet and ten-epoch experiment. No saved outputs are embedded. |
| `ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/example_physionet.py` | Distribution example: WFDB waveform loading, SCP diagnostic-superclass aggregation and folds 1–9 versus 10. Uses a placeholder dataset path and is not a trained project model. |
| `ecg files/` | Expanded four-class image cohort. Raw folders include `Normal`, `MI`, `Abnormal`, `HIstory of MI`. Audit counts, duplicate groups and exclusions are defined by the new dataset audit, not manuscript totals. |
| `mendely images.zip` | Source image archive with 932 ZIP entries: 928 JPG entries and four directory entries. It contains no model code. Uncompressed entry total is 615,232,522 bytes; archive was not duplicated. |
| `ECG_Project_Logbook.docx` | Extracted all 40 Word paragraphs, including the table. Dataset validation, saved grouped splits, train-only preprocessing, experiment logging and model progression are requested. LSM and convolution approach are explicitly undefined placeholders. Ongoing experiment table is blank. |
| `ECG_Simple_To_Do.xlsx` | Read `To Do` sheet, used range A1:E18 via OOXML. Rows 5–16 are tasks, not measurements; A2 and A18 describe the earlier publication-first plan. No model results or formulas were found in the inspected cells. |
| `ECG_Project_Presentation.pptx` | Extracted all ten slides. Slides 3–5 describe the old extraction/model; slide 7 lists limitations; slide 8 says project result TBD; slide 10 calls for actual execution. The private-Swiss-source assertion conflicts with later manuscripts. |
| `proj desc.pdf` | Two pages. Original project objective is direct classification of scanned ECG images. It describes planned public data and model development, not completed experiments. |
| `report april 9 .pdf` | Fourteen pages. Early draft reports ten-epoch 1D ResNet results (86.02% overall accuracy; 87.13% high-quality subset) and source counts totaling 928. Numeric and narrative inconsistencies require original predictions before reuse. |
| `ecp google doc.pdf` | Nineteen pages. Intermediate domain-shift draft mixes the 94.62% enhanced result with old architecture/error text and calls a decrease to 94.15% an improvement. It also reports an older 31.43% external fine-tuning result. |
| `ecg_manuscript_corrected_v2.zip` | Nineteen entries, including `ecg_manuscript/main.tex`, `references.bib`, fourteen PNG figures and one PDF figure. Reports a different PTB-XL fine-tuning split from v3. No executable training code. |
| `ecg_manuscript_corrected_v3.zip` | Nineteen entries under `manuscript_v2/ecg_manuscript/`. `main.tex` Methods, Results, Limitations and tables document the 94.62% source claim, enhanced four-block architecture, 600-image PTB-XL evaluation and 502/53/45 folds. Comments retain missing-evidence TODOs. No executable training code. |
| `ECG_Paper___Miriam.pdf` | Seventeen pages. Later paper broadly follows v3 and lists the project GitHub repository on page 6. Its reported results remain historical manuscript claims until linked to run artifacts. |
| `AI_Transforms_Paper_ECGs___Methodology_paper.pdf` | Twenty-four-page literature-review manuscript, titled *Artificial Intelligence Transforms Paper ECGs into Scalable Digital Diagnostics*. Context and literature summary; not evidence that a local model was trained. |
| `physionet paper.pdf` | Four-page external research paper, *Combining Hough Transform and Deep Learning Approaches to Reconstruct ECG Signals From Printouts*, DOI 10.22489/CinC.2024.097. SignalSavants digitization method and challenge result; not the user's implementation or result. |
| `ECG_from_paper_to_diagnosis.pdf` | Seventeen pages. Despite filename, its actual title is *ECG Sonification Methods for Robust and Generalizable Clinical Decision Support*. It reviews auditory ECG methods and is peripheral to current image baselines. |
| `architecture.png`, `architecture_overview.png` | Existing loose architecture illustrations. Preserve as previous work; diagram files cannot validate a training run. |

## Remote references

The local September paper's Data Availability section points to [the project GitHub repository](https://github.com/itsmiri-eng/ecg-image-domain-shift). Its current source and artifacts could not be fetched in this session, so this audit is explicitly limited to the available local project. The URL's appearance in a PDF does not establish that the local files match a repository revision.

The manuscripts identify the Mendeley image source as DOI [10.17632/gwbz3fsgp8.2](https://doi.org/10.17632/gwbz3fsgp8.2). The source attribution is recorded from the existing papers; the new local archive and image hashes provide the reproducible identity of the files actually used. Do not infer patient identity or diagnostic validity from folder names alone.

The [official Mendeley v2 page](https://data.mendeley.com/datasets/gwbz3fsgp8/2), checked in this session, attributes the dataset to Ch. Pervaiz Elahi Institute of Cardiology, Multan, Pakistan and identifies a CC BY 4.0 license. This supports the later manuscripts' source attribution; it does not validate each local label or patient identity. The archive check independently establishes that all 928 local source JPEGs match the retained ZIP byte-for-byte.

Method references checked: [scikit-learn LDA and shrinkage](https://scikit-learn.org/stable/modules/lda_qda.html) and [training-only preprocessing/data leakage](https://github.com/scikit-learn/scikit-learn/blob/main/doc/common_pitfalls.rst). Shrinkage regularizes covariance estimation; fitting PCA only on training samples prevents test information entering this learned projection.

The PTB-XL package is identified locally as version 1.0.3, with its own metadata, checksums, license and changelogs. See `ptbxl_secondary_audit.md` for the separate waveform dataset assessment. Its diagnostic taxonomy is not interchangeable with the four image-folder labels.

## Reading method and limits

PDF text was extracted page by page with the bundled `pypdf` runtime; layout extraction was additionally used for the April and Google-doc reports because basic extraction split words across many lines. DOCX, PPTX and XLSX text was read from their OOXML containers; all slide text and all task-table cells were inspected. LaTeX inside both manuscript archives was inspected directly, including comments omitted from rendered PDFs. This was a scientific content and implementation audit, not a layout-quality review or an independent clinical review. Bibliographic claims and published performance attributed to third parties were not independently revalidated because they are not required to execute the two local baselines.
