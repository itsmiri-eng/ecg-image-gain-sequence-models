# Local dataset inventory and integrity audit

Audit date: 2026-09-23. Original files were read without modification. Counts below describe the local files, rather than the nominal counts in dataset folder names or manuscript prose.

## Primary project dataset: scanned ECG pages

The extracted `ecg files/` directory and `mendely images.zip` contain the same **928 source JPEG pages**. The archive contains four directory entries plus these 928 images. Every archive image has the same filename and SHA-256 file hash as its corresponding extracted source image; none is missing, changed, or additional. The archive is a second copy of the same data, not an independent evaluation set. See `data/audit/archive_provenance.json` for the independently reproduced archive comparison and archive hash.

| Local directory / label | Source JPEG pages | Unique file hashes | Unique decoded pixel images | Redundant copies |
| --- | ---: | ---: | ---: | ---: |
| `Abnormal` | 233 | 233 | 233 | 0 |
| `HIstory of MI` | 172 | 86 | 86 | 86 |
| `MI` | 239 | 30 | 30 | 209 |
| `Normal` | 284 | 142 | 142 | 142 |
| **Total** | **928** | **491** | **491** | **437** |

All source images load successfully with Pillow, decode as RGB JPEG, and have dimensions **2213 × 1572 pixels**. No corrupt or unreadable JPEG was found. Pixel equality was checked using the decoded dimensions, color mode, and pixel bytes; it did not reveal additional duplicate groups beyond file-byte equality. No exact file or decoded-pixel duplicate group spans different class directories.

The directory also contains `MI/.ipynb_checkpoints/MI(1)-checkpoint.jpg`, an exact copy of `MI(1).jpg`. This explains the recursive total of 929 JPEG files; the checkpoint is excluded from the source count. Two `.DS_Store` files are irrelevant filesystem metadata. The main MI directory has 239 images because `MI(215).jpg` is absent; the archive also lacks this file. The name `HIstory of MI` contains a capital `I` and must be handled explicitly on case-sensitive systems.

The 491 unique byte groups comprise 233 singletons, 228 pairs, 29 groups of eight, and one group of seven. The duplication pattern is systematic: Normal page numbers repeat after an offset of 142; History-of-MI page numbers repeat after 86; MI page numbers generally repeat in offsets of 30. These are unchanged copies, not distinct acquisitions or independently generated augmentations. A random split of filenames would permit identical pages to enter both training and evaluation data.

## Identity, layout, and leakage limits

Visual inspection of one page from each class shows a report header containing a printed numeric `ID`, three rows of four lead panels, and a fourth rhythm-strip row. Printed IDs are therefore present in the image pixels. There is no separate patient-identity table accompanying this local image collection. An extracted printed ID must be described as an **unverified report/patient identifier**, and should be used conservatively to group related records. Equality or inequality of these printed IDs alone does not prove patient identity or patient independence.

Exact copies necessarily repeat their printed header IDs. Additional repeated IDs among distinct pages must be determined from the separately implemented header audit; this initial inventory does not invent or infer their counts. The final `data/audit/header_ids.csv` and its associated extraction evidence, when generated, are the authoritative mapping. Header extraction should use only image content and must preserve uncertainties and any manual corrections explicitly. Unresolved IDs require a stated exclusion or conservative grouping policy before splitting.

A classification image should exclude the report header and footer to reduce dependence on printed identifiers and acquisition metadata. It must retain the actual page layout. Treating the whole page as 12 horizontal bands would not match the observed three-by-four lead arrangement plus rhythm strip, and would not establish valid waveform digitization.

The separate conservative similarity audit screened 491 unique byte images using a fixed trace-region crop. Its stored result contains 1,508 candidate pairs and zero links meeting both stringent correlation and mean-absolute-difference thresholds; see `data/audit/near_duplicate_summary.json`. This is a limited similarity screen. It does not prove that shifted, rescaled, redrawn, or repeated-acquisition ECGs are absent, or that patients are independent. The full split must also honor any repeated printed identifiers and identical model-input tensors.

## OCR environment attempts

A Swift/Vision attempt could not start because the installed Xcode tooling requires license agreement. No license acceptance was performed. A second attempt dynamically loaded macOS Vision through the installed PyObjC bridge. The recognition request returned failure with no populated error or result for both CPU/default execution and URL/data input in the sandbox. No IDs were accepted from these failed attempts and no header CSV was written by that helper.

An escalation request for a bounded two-image Vision probe was interrupted by the coordinator while switching to a sandboxed template-recognition alternative. It was not a successful OCR run. The optional `src/preprocessing/ocr_header_ids.py` helper records the attempted Vision method but is not the method used for the final offline ID mapping. There is no need to accept the Xcode license or grant additional permissions to use the alternative audit.

## Secondary dataset: PTB-XL v1.0.3

The local PTB-XL metadata contains 21,799 distinct ECG records from 18,869 patients. Each record has a 12-lead, 10-second signal in both 100-Hz and 500-Hz forms: 21,799 header/data pairs per rate, or 43,598 `.hea` plus 43,598 `.dat` files overall. All metadata-referenced files exist. Every 100-Hz header specifies 1,000 samples and every 500-Hz header specifies 5,000; observed data byte sizes match 12-channel signed 16-bit storage.

The supplied patient-level fold metadata yields the following partitions, with zero patients appearing in multiple folds or partitions:

| Official partition | Records | Patients |
| --- | ---: | ---: |
| Folds 1–8, training | 17,418 | 15,023 |
| Fold 9, validation | 2,183 | 1,942 |
| Fold 10, test | 2,198 | 1,904 |

PTB-XL diagnostic superclass counts are overlapping multilabel counts: NORM 9,514; MI 5,469; STTC 5,235; CD 4,898; HYP 2,649. There are 411 records with none of these diagnostic superclasses. These labels are not equivalent to the four mutually exclusive image-folder labels. The two sampling rates are alternate representations of the same ECG, not independent samples. No local provenance link establishes a patient or record relationship between these waveforms and the Mendeley scanned pages.

The expanded audit in `data/audit/ptbxl_inventory.json` verifies all 87,203 supplied checksum entries, including waveform integrity against header fields, and finds no exact same-rate waveform duplicate groups. See `docs/ptbxl_secondary_audit.md` for the detailed secondary-dataset findings and limitations.

## Dataset recommendation

Continue the existing four-class image project using the deduplicated Mendeley pages as the primary dataset, with printed-ID groups and all other verified duplicate relationships fixed before splitting. Count the effective 491 unique pages and only 30 unique MI pages explicitly; do not describe 928 filenames as independent records or patients. Interpret evaluation as a small exploratory image-classification study, with uncertainty and identity limitations reported.

Keep PTB-XL as a separate, unused secondary dataset for the current image baseline. Its explicit patient metadata and official folds support a future waveform or rendered-waveform study, but using it would require a separate label definition, preprocessing protocol, and experimental scope. Combining the two datasets would not constitute a valid cross-dataset validation without resolving their different input representations and labels.
