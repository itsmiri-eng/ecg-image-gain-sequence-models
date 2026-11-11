# Ideas to make the paper stronger (saved 11 November 2026, not started)

Miriam pasted this list on 11 Nov 2026 and asked to save it without acting on it yet. The
"Where it stands" notes were added by Claude after checking the project folder; nothing was run.

## 1. Test realistic image degradation

Make controlled versions of the ECG page images with rotation, noise, blur, compression, grid
variation and perspective distortion. Check whether fixed-gain training stays beneficial. This
addresses the biggest gap between the clean rendered pages and real ECG photos or scans.

**Where it stands.** Not done for this study. A degradation experiment (rotation, blur, brightness)
exists only in the *older* paper that used to be on Overleaf (backup:
`paper/overleaf_backup_2026-11-10_before_replace/main.tex`, section "Robustness to image
degradation", figure `degradation_experiment.png`). That was a different pipeline (binary-mask
extraction, not PTB-XL pages and not these models), so its numbers cannot be reused here. No
degradation code exists in `src/`. `paper/PUBLICATION_PLAN.md` already notes that such a test on
held-out patients is "a controlled stress test, not clinical deployment validation". If done, it
needs a frozen protocol amendment first and must not re-run the fold-10 test inference of existing
runs (degraded copies would be a new, pre-specified evaluation).

## 2. Improve external statistical inference

Use a patient-cluster bootstrap where patient IDs exist, account for repeated recordings of the
same patient, and report per-class uncertainty.

**Where it stands.** PTB-XL already uses a patient-cluster bootstrap. On the external cohorts,
records are resampled because Georgia has no patient IDs (`docs/external_test_plan.md`, section 6);
the China cohort has not been checked yet for usable patient IDs. Per-label external AUROC is in
the Supplementary Information; whether it carries per-label confidence intervals was not checked.

## 3. Finish the 40-epoch and pretrained ResNet-18 experiments

These test whether the architecture conclusions depend on training length or on a stronger
image encoder.

**Where it stands.** ResNet-18 is done and in the paper (`docs/PROJECT_STATUS.md`, update of
11 Nov evening). The 15 runs of the 40-epoch sensitivity are in progress under
`docs/protocol_amendment_2026-11-11.md` (results folder `results/ptbxl_v2_long40/`).

## 4. Investigate the mechanism behind the gain effect

There is evidence that global models read the calibration pulse as a voltage-scale reference,
while local models gain more from removing adaptive-gain variation. Quantify this systematically,
ideally with controlled interventions and analyses specified in advance.

**Where it stands.** A first intervention exists: the pulse-swap experiment
(`docs/secondary_analyses_protocol.md`, section D; results in
`results/secondary/pulse_swap/summary.md`, validation fold, seed 42 only). Extending it to all
seeds and to the fixed-gain models, under a new frozen protocol, would be the natural next step.
