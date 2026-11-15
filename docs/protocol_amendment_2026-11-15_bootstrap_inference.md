# Statistical revision, 15 November 2026: finite-resample bootstrap p-values and Holm correction

Written and frozen on 15 Nov 2026 **before** any statistic was recomputed and before any result of the
fixed-gain capacity-matched runs (amendment `docs/protocol_amendment_2026-11-15_fixedgain_wide.md`) existed.
It changes **how p-values are computed from the bootstrap**, for every comparison in the paper. It does not
change any model, prediction, hypothesis, comparison family, test statistic or bootstrap design.

## What was wrong

Every comparison uses the same routine (`src/external_test.py::hierarchical_difference`): 2,000 hierarchical
bootstrap replicates Δ* of the difference in mean-over-seeds AUROC, a percentile 95% CI, and

    p_old = 2 · min( #{Δ* ≤ 0} / B , #{Δ* ≥ 0} / B ),   B = 2,000.

1. **p_old can be exactly 0.** When no replicate crosses zero the estimate is 0, and the Holm step
   (multiplication by the number of remaining hypotheses) leaves it at 0. The tables printed this as
   "p < 0.001" and the Methods said "an adjusted p < 0.001 means that no bootstrap replicate crossed zero".
   That statement is not supported by 2,000 resamples: a Monte Carlo p-value from B resamples cannot be shown to
   be smaller than about 1/(B+1) per tail. With B = 2,000 the smallest supportable two-sided p is
   2/2,001 ≈ 0.001, and after Holm correction over 10 comparisons it is ≈ 0.01 (0.008 over 8 comparisons).
   All adjusted "p < 0.001" entries were therefore over-stated by up to an order of magnitude. This was raised
   in the supervisor's review of 15 Nov 2026.
2. The resolution was never stated.

## Revised procedure (applies to every comparison)

- **Resamples:** B = 10,000 (seed 2024). The first 2,000 replicates are the same random draws as before, so the
  old numbers must be reproduced exactly from them; this is checked for every comparison and recorded.
- **p-value:** the finite-resample (add-one) estimator, which can never be zero
  (Davison & Hinkley, *Bootstrap Methods and their Application*, 1997, ch. 4; Phipson & Smyth,
  Stat. Appl. Genet. Mol. Biol. 2010;9:39):

      p = min{ 1, 2 · min( (1 + #{Δ* ≤ 0}) / (B + 1) , (1 + #{Δ* ≥ 0}) / (B + 1) ) }.

  This is the p-value obtained by inverting the percentile interval, so "p < 0.05" and "the 95% percentile CI
  excludes zero" agree up to Monte Carlo error.
- **Holm correction:** unchanged step-down procedure, within the same families as before (10 head pairs; 8
  fixed-versus-adaptive comparisons; the families of the dated sensitivity analyses). Applied to the revised p.
- **Resolution, always reported:** the smallest attainable raw p is 2/10,001 ≈ 0.0002. The smallest attainable
  Holm-adjusted p in a family of m comparisons is m × 0.0002 (0.002 for m = 10; 0.0016 for m = 8; 0.001 for
  m = 5; 0.0008 for m = 4; 0.0004 for m = 2). A value at this floor is printed as "≤" that value and means only
  that none of the 10,000 replicates crossed zero; no smaller p is claimed.
- **Confidence intervals:** percentile 95% intervals from the 10,000 replicates (they may differ from the old
  ones in the third decimal by Monte Carlo error).
- **Sensitivity check (reported in the supplement, not used for claims):** a null-centred bootstrap p,
  p_c = (1 + #{ |Δ* − Δ̂| ≥ |Δ̂| }) / (B + 1), which tests H0: Δ = 0 by shifting the bootstrap distribution to
  the null (Hall & Wilson, Biometrics 1991;47:757).

## Wording rule for the manuscript (fixed now)

- A difference is called **statistically supported** only if its Holm-adjusted p < 0.05 (equivalently its
  interval, after correction, is clearly away from zero). Otherwise it is reported as a **point estimate**
  with its interval, and words such as "improved", "better", "outperformed" or "never worse" are not used for it.
- No non-inferiority or equivalence claim is made anywhere: no margin was prespecified and no such test was done.
  External hypothesis H2 was worded "not worse" in the frozen plan; it is evaluated as a two-sided
  superiority test, as it always was computed, and is now described that way.

## Limitations that remain (stated in the paper)

- The top level of the hierarchy has only five training seeds (three in the dated sensitivity analyses). A
  bootstrap over so few units cannot represent between-seed variance well, so intervals are likely somewhat too
  narrow and p-values somewhat too small. The p-values are approximate.
- The external cohorts have no patient identifiers, so records are resampled; repeated ECGs of one patient
  would make those intervals too narrow.
- The percentile-inversion p-value is an approximation, not an exact test.
- PTB-XL test-fold comparisons remain exploratory regardless of their p-values.
- This revision was made after the results were known. It can only weaken, not strengthen, the earlier
  p-values at the floor; other p-values change by Monte Carlo error only.

## Record keeping

- The old outputs are archived byte-for-byte, with SHA-256 hashes, in
  `results/stats_legacy_2000resamples_20261115/` before anything is regenerated, and remain in the Git history.
- The routine keeps the old formula as `p_legacy_2000` in every output row, so each table can be read both ways.
- The Supplementary Information gets a table of every comparison whose adjusted p changed category
  (old versus revised).
- The frozen external plan (`docs/external_test_plan.md`) specified 2,000 resamples. Hypotheses, cohorts,
  labels, families and the test statistic are unchanged; only the p-value estimator and the number of resamples
  differ, and both versions are reported.
