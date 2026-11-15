# Protocol note, 15 November 2026: two descriptive additions (Additional analysis)

Written on 15 Nov 2026 before either quantity was computed. Both add uncertainty estimates to differences whose
**point estimates are already in the paper** (Tables 1 and 4), so the direction of each is known; nothing is
trained and no inference is run. They respond to the supervisor's review (points 6 and 8), which asked that fixed
gain's lower PCA + LDA macro AUROC be acknowledged and that the signal-versus-image gap be reported separately
for the compact models and for ResNet-18. Both are post hoc and are labelled **Additional analysis**.

1. **PCA + LDA, fixed minus adaptive gain.** Runs `ptbxl_v2_fg_lda_seed42` and `ptbxl_v2_lda_wide_seed42` (the two
   single fits reported in the paper). Macro AUROC and HYP AUROC on the PTB-XL test fold (patient-cluster
   bootstrap) and on each external cohort (record bootstrap), 10,000 resamples, Holm over the 2 metrics within each
   cohort. There is no training seed, so the interval reflects test sampling only.
2. **Image model minus signal reference.** For each image network and page type (five heads and ResNet-18, adaptive
   and fixed gain: 12 groups), the difference in mean-over-seeds macro AUROC from the 1D ResNet signal reference,
   on the PTB-XL test fold and each external cohort, with the hierarchical bootstrap (seeds 42–46; 42–44 for
   ResNet-18) and Holm over the 12 comparisons within each cohort. Seeds of different architectures are paired only
   by their number, as in the head-pair comparisons.

Interpretation rule: a gap is described as statistically supported only if its Holm-adjusted p < 0.05; a gap whose
interval includes zero is described as "not distinguishable in this sample", not as equality (no equivalence margin
was prespecified).
