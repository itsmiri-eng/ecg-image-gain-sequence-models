# Interrupted runs: Windows user sign-out on 5 Nov 2026 at 16:18 local (moved 6 Nov 2026)

Four runs of the capacity-matched head (`ptbxl_v2_cnn_pool_wide_seed{42,43,44,46}`) were in epochs 5–7 of 20.
They were running on 2 CPU cores at the user's request. At 16:18 local on 5 Nov the Windows log shows a user
logoff and then sleep:

- Winlogon 7002, "User Logoff Notification";
- Winsrv 10001, where Steam and another program tried to veto the shutdown;
- Kernel-Power 42, sleep.

Signing out ends every process of that account, so the runs stopped. The computer did not restart; it was
last booted on 26 Oct.

None of the four froze a checkpoint or started test inference. There is no `selection_frozen.json` and no
`test_inference_state.json`, and there are no experiment-log rows. Their partial `results/` and `models/` folders,
their logs and the 5 Nov lane logs were moved here, unedited, so the run IDs can be retrained from scratch under
the same frozen config. The four seed-45 runs completed before the sign-out are unaffected.

These files are not evidence for any reported result. This is probably also the cause of the 3 Nov loss
(`../2026-11-03_paused_then_closed/`). From 6 Nov, lanes run with hidden windows. The user has been told
that a sign-out stops training and that switching user does not.
