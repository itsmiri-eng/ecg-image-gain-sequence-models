# Protocol amendment, 11 November 2026: signal-reference seeds 45 and 46

Written **before** either run below started and before any of their results existed
(2026-11-10 20:55 UTC = 2026-11-11 00:55 local time). Nothing below changes a completed run, a hyperparameter,
a config used by an existing run, or the frozen external-test analysis.

## Reason

The image networks were trained with seeds 42–46 (`docs/protocol_amendment_2026-11-03.md`), but the
signal-domain reference (`experiments/configs/ptbxl_v2_signal_resnet1d.json`) only with seeds 42–44, because
that amendment did not cover it. A reviewer-style read of the manuscript (11 Nov) flagged the mismatch. The two
seeds are added only so that every trained network in the paper reports the same seeds.

## A. Additional runs

- `ptbxl_v2_signal_resnet1d_seed45` and `ptbxl_v2_signal_resnet1d_seed46`.
- Same frozen config (`experiments/configs/ptbxl_v2_signal_resnet1d.json`, unchanged; its `seeds` field
  records the original plan and is not edited), same runner (`src/run_ptbxl_signal.py`, unchanged), same
  unchanged shared trainer, same device (the PC GPU, `cuda`, as for seeds 42–44), same command as seeds 42–44:
  `python -X utf8 -m src.run_ptbxl_signal --seed <seed>`.
- One test inference per run, as for every run. A failed run is reported as failed and is not rerun under the
  same ID; an interrupted run is moved to `experiments/abandoned_runs/` and restarted from scratch.

## B. Reporting on PTB-XL

- The signal reference is reported as mean ± SD over seeds 42–46 wherever the image models report seeds 42–46.
  The per-seed values of all five seeds are reported in the Supplementary Information, so the earlier
  three-seed summary (0.928 ± 0.001) stays recoverable.
- The paired signal-versus-best-image comparison at seed 42 is unchanged (it uses seed 42 only).

## C. External test (descriptive sensitivity analysis)

- The frozen external plan (`docs/external_test_plan.md`, §5) includes "every completed run", which at the
  freeze meant signal seeds 42–44. It is **not** edited.
- Seeds 45/46 get exactly one inference per external dataset (Georgia; Chapman-Shaoxing + Ningbo), with the
  same pages/signals cache, the same 100 Hz resampling and the same one-inference guard
  (`inference_state.json`). Before inference, each reloaded model must reproduce its saved PTB-XL validation
  probabilities (tolerance 1e-3), as for all other runs.
- `src/external_test.py` refuses to start inference while any run already has a state file. A new step
  (`infer-added`) runs only completed runs that have no state file yet and refuses to touch any other run. No
  existing prediction file is rewritten.
- These two runs are a **descriptive** addition. The confirmatory hypotheses H1 and H2 involve only the image
  heads and are unchanged; re-running `analyze` must reproduce `hypotheses.csv` exactly, which is checked.
  The signal reference stays a secondary, descriptive comparison, as in the frozen plan.
