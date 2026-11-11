# Interrupted runs: 40-epoch sensitivity, seed 42 (11 Nov 2026)

Runs: `ptbxl_v2_long40_cnn_pool_seed42`, `ptbxl_v2_long40_cnn_pool_wide_seed42`, `ptbxl_v2_long40_hybrid_vit_seed42`
(amendment `docs/protocol_amendment_2026-11-11.md`, part A).

## What happened

- **Start and pause.** All three started at 10 Nov 21:05 UTC. Each was suspended (paused) at about 21:10 UTC,
  during epoch 1 of 40, so that the two ResNet-18 runs could use the GPU.
- **Restart.** The PC restarted on 11 Nov at about 03:46 UTC (System log: event 1074 "restart" and 6006
  "event log stopped" at 07:46 local time), which ended the suspended processes. The queued runs behind them
  in their lanes never started and created no files.
- **How far they got.** None of the three reached checkpoint selection or test inference. There is no
  `best.pt`, `selection_frozen.json` or `test_inference_state.json`, and no experiment-log row.

## What was done

- **Moved, not edited.** Their partial files (results and models folders, logs) were moved here unchanged on
  11 Nov at about 08:10 UTC.
- **Restarted from scratch.** The three runs were started again under the same run IDs, with the same configs.
  This follows the 1, 3, 5 and 6 Nov practice for runs interrupted before any test inference (see the other
  folders here).
- **Not evidence.** Nothing in this folder supports any reported result.
