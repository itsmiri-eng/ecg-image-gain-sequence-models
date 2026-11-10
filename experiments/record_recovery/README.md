# Run records written after the fact (2 Nov 2026, Windows PC)

## What happened

The first GPU queue on the Windows PC (`src.run_queue --device cuda`, started 1 Nov 2026 19:58 UTC)
ran 12 runs. Every one trained, froze its checkpoint, ran its single test inference and saved all
predictions, metrics and the patient bootstrap. Each then stopped at the last step, before writing
its experiment-log row and `run.json`, with:

    ValueError("Protected baseline artifacts changed: ['experiments/experiment_log.csv']")

Cause: `src/baseline_preservation.py` built the log's name with `str(LOG.relative_to(ROOT))`, which
is `experiments\experiment_log.csv` on Windows. The lock file keys it as
`experiments/experiment_log.csv`, so the check treated the (legitimately growing) log as a changed
protected file. The error was raised before anything was appended, so the log was not touched. The
Mac never showed this because it uses `/`.

## What was done

1. Fix: `log_name = LOG.relative_to(ROOT).as_posix()` (line 14). Nothing else in that file changed.
   The 12 runs' `source_snapshot/src/baseline_preservation.py` keep the original version they ran with.
2. `src/recover_ptbxl_run_record.py` rebuilt each log row and `run.json` from the run's saved files,
   with the same code path as `src/run_ptbxl_experiment.py`. No model was loaded, the test partition
   was not read again, and no test inference was repeated.
3. Faithfulness check: rebuilding the completed Mac run `ptbxl_v2_cnn_pool_seed42` reproduced its real
   log row and `run.json` exactly (`--check-against`, zero differences; timing keys excluded).
4. Before writing, each run had to show: `failure.json` with exactly the error above, a completed
   `test_inference_state.json` with one inference call, identical checkpoint hashes in the freeze,
   the test-inference record and `best.pt`, a saved bootstrap, and no existing log row or `run.json`.
5. Each `failure.json` was moved (not deleted) to `<run_id>/failure.json` here, next to
   `<run_id>/recovery.json` (the exact log row and `run.json` written).

## What differs from a normal record

- `run.json` has an extra `record_recovery` block saying all of the above.
- `completed_utc` is the modification time of `failure.json` (when the original process stopped), and
  `total_runtime_seconds` is the runtime that process recorded in `failure.json`.
- The log row's `notes` end with a sentence saying the record was written afterwards.

Verified after writing: `python -m src.baseline_preservation` passes (66 protected files unchanged,
original log prefix preserved), the pre-recovery log is an exact byte prefix of the new one with 12
rows appended, and `python -m src.status` reports 13 of 13 planned runs finished.

| Run | Test macro AUROC |
|---|---|
| ptbxl_v2_cnn_pool_seed43 | 0.8272 |
| ptbxl_v2_cnn_pool_seed44 | 0.8255 |
| ptbxl_v2_hybrid_vit_seed43 | 0.8883 |
| ptbxl_v2_hybrid_vit_seed44 | 0.8856 |
| ptbxl_v2_temporal_conv_seed43 | 0.8791 |
| ptbxl_v2_temporal_conv_seed44 | 0.8788 |
| ptbxl_v2_cnn_lstm_seed43 | 0.8807 |
| ptbxl_v2_cnn_lstm_seed44 | 0.8862 |
| ptbxl_v2_fg_cnn_pool_seed42 | 0.8537 |
| ptbxl_v2_fg_cnn_lstm_seed42 | 0.8842 |
| ptbxl_v2_fg_temporal_conv_seed42 | 0.8956 |
| ptbxl_v2_fg_hybrid_vit_seed42 | 0.8986 |

## Still open

The test suite on Windows: 197 of 219 pass. The 22 failures predate this fix and are Windows
portability problems in tests and verifiers (no `git` on the PC, symlinks need Windows Developer
Mode, a Mac absolute path in `tests/test_portable_cache.py`, and `str(path.relative_to(...))` with
backslashes in the `verify_ptbxl_*` scripts). They do not affect the training code path above.
