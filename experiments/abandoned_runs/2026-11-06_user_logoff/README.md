# Interrupted runs: Windows user sign-out on 6 Nov 2026 at 21:53 local (moved 7 Nov 2026)

The Windows System log shows a user logoff at 21:53:17 local (Winlogon 7002). A program then requested sleep
(Kernel-Power 187), the system slept (42) and resumed 4 s later (107). The computer did not restart; it was
last booted on 26 Oct. A sign-out ends all processes of the account, so four runs stopped:

- `ptbxl_v2_cnn_lstm_seed46` (epoch 17)
- `ptbxl_v2_cnn_pool_seed46` (epoch 12)
- `ptbxl_v2_fg_cnn_pool_seed45` (epoch 6)
- `ptbxl_v2_fg_hybrid_vit_seed45` (epoch 6)

None of them froze a checkpoint or started test inference. There is no `selection_frozen.json` and no
`test_inference_state.json`, and there are no experiment-log rows. Their partial `results/` and `models/`
folders, their logs and the 6 Nov lane logs were moved here, unedited, so the run IDs can be retrained from
scratch under the same frozen configs.

These files are not evidence for any reported result. This is the same cause as
`../2026-11-05_user_logoff/` (and probably `../2026-11-03_paused_then_closed/`).
