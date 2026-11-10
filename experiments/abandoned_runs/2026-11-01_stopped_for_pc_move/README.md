# Stopped run: ptbxl_v2_cnn_pool_seed43 (1 Nov 2026)

Stopped by hand (SIGKILL) at epoch 7 of 20, after being paused for power from about 20:35 local,
because the project moved to an NVIDIA PC. It never reached checkpoint freezing or test inference
(no `selection_frozen.json`, no `test_inference_state.json`) and wrote no experiment-log row.

Its partial files were moved here so the canonical run ID can be trained from scratch on the PC.
They are not evidence for any reported result. Nothing in this folder was edited.

The queued runs that had not started (seeds 43/44 of every network, cnn_pool seed 44, and the
fixed-gain network retrains) were cancelled at the same time; none had created files.
