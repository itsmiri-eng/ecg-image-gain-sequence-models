# Domain Shift and Format Sensitivity in ECG Image Classification Pipelines

An experimental biomedical engineering and machine-learning project investigating whether ECG image-classification pipelines remain reliable when applied across different datasets, rendering formats, grid styles, and image-quality conditions.

## Project Overview

Many electrocardiograms remain available as printed records, scanned documents, photographs, or raster-image exports rather than native digital signals. Machine-learning models cannot directly process these records without first extracting and reconstructing the ECG waveforms.

This project developed an end-to-end pipeline that converts paper-style 12-lead ECG images into calibrated one-dimensional signals and classifies them into four diagnostic categories:

- Normal
- Myocardial infarction
- Abnormal
- History of myocardial infarction

The main focus of the study was not only classification accuracy, but also whether performance remained stable when the image source and formatting changed.

## Research Question

Can an ECG image-processing pipeline perform well within its original dataset while remaining robust to changes in image format, rendering conventions, grid appearance, blur, rotation, brightness, and dataset source?

## Pipeline

The experimental workflow included:

1. ECG image loading and standardization
2. Computational image-quality estimation
3. Grid and calibration-pulse detection
4. Physical amplitude calibration using the standard 10 mm/mV reference
5. ECG trace isolation using thresholding and contour analysis
6. Morphological skeletonization using Zhang–Suen thinning
7. Fixed 3 × 4 segmentation into 12 lead regions
8. Reconstruction and resampling of each lead to 1,000 samples
9. Formation of a 12 × 1,000 model input
10. Classification using a one-dimensional residual neural network
11. Ablation testing of alternative extraction methods
12. Robustness testing under rotation, blur, and brightness changes
13. Cross-dataset evaluation using PTB-XL-derived ECG images

## Dataset

The within-distribution experiments used the Mendeley ECG Images Dataset of Cardiac Patients, containing 928 twelve-lead ECG images across four classes.

A stratified 80:20 train-test split was used:

- Training images: 742
- Test images: 186

The external evaluation used 600 ECG images generated from PTB-XL signals using ECG-Image-Kit.

## Model

The reconstructed signals were classified using a 1D-ResNet architecture with:

- 12 input channels
- Four residual stages
- Global average pooling
- Dropout
- Four-class output layer
- Class-weighted cross-entropy loss
- Adam optimization
- Cosine-annealing learning-rate scheduling

## Main Results

The final pipeline achieved the following performance on the held-out within-distribution test set:

- Accuracy: 94.62%
- Matthews Correlation Coefficient: 0.9287

However, zero-shot evaluation on differently rendered PTB-XL images produced approximately 25–26% accuracy, close to chance level for a four-class problem.

Partial fine-tuning improved accuracy to 31.43%, but did not eliminate the domain gap.

## Key Finding

High performance within a controlled dataset does not necessarily demonstrate real-world robustness.

The pipeline was strongly affected by:

- ECG grid morphology
- Image-rendering conventions
- Grayscale relationships between the grid and waveform
- Fixed page-layout assumptions
- Rotation
- Blur
- Thresholding behavior
- Diagnostic-label mapping

These results demonstrate why biomedical AI systems should be evaluated across datasets and acquisition formats rather than assessed using only a single-source train-test split.

## Robustness Analysis

The pipeline was particularly sensitive to geometric and structural image changes.

### Rotation

- ±5°: 30.65% accuracy
- ±15°: 22.58%
- ±30°: 26.34%

### Gaussian Blur

- Kernel size 3: 52.15%
- Kernel size 7: 26.88%
- Kernel size 15: 27.96%

Brightness changes were less damaging than rotation and blur.

## My Contribution

I carried out the experimental development and evaluation of the ECG image-processing and classification pipeline under the supervision of Dr. Mohamed Elgendi at Khalifa University.

My work included:

- Developing and refining the ECG image-processing workflow
- Implementing trace extraction and waveform reconstruction
- Evaluating calibration and extraction approaches
- Training and assessing the classification model
- Conducting ablation experiments
- Designing synthetic image-degradation tests
- Performing cross-dataset evaluation
- Analyzing domain-shift and format-sensitivity failure modes
- Contributing to the preparation of the research manuscript

Formal manuscript contribution statements are still being finalized.

## Current Status

The experimental work is complete, and the research manuscript is currently being finalized for submission.

This repository presently provides a public overview of the project. Source code, manuscript content, and additional results will only be released if approved by the research team.

## Supervision

Conducted at Khalifa University under the supervision of:

**Dr. Mohamed Elgendi**  
Department of Biomedical Engineering and Biotechnology  
Khalifa University

## Important Disclaimer

This project is an experimental research study and is not a clinically validated diagnostic system. Its results should not be interpreted as evidence of clinical readiness or universal diagnostic performance.
