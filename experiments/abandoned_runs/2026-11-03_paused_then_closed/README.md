# Interrupted runs: paused on 3 Nov 2026, processes closed before resuming (moved 5 Nov 2026)

Five GPU runs started at 10:33 UTC on 3 Nov under `docs/protocol_amendment_2026-11-03.md`:
`ptbxl_v2_{cnn_pool, cnn_lstm, temporal_conv, hybrid_vit}_seed45` and `ptbxl_v2_cnn_pool_wide_seed42`.

- **Pause.** At 11:42 UTC (15:42 local) all were frozen in place with `experiments/pause_resume.ps1` so that
  the PC could be used for something else. Each was in epoch 15 of 20.
- **Closure.** By 5 Nov all frozen Python processes had been closed. The computer had not restarted. They
  never resumed.
- **What they never reached.** None of them froze a checkpoint or started test inference. There is no
  `selection_frozen.json` and no `test_inference_state.json`, and none wrote an experiment-log row.
- **What was moved.** Their partial `results/` and `models/` folders and their logs were moved here, unedited,
  so the run IDs can be trained from scratch under the same frozen configs. The queue logs of 3 Nov are here too.
- **Evidence status.** These files are not evidence for any reported result.
- **Downloads.** The external-data downloads paused at the same time keep the files they had already
  verified. The downloader re-checks every file's SHA-256 when it resumes.
