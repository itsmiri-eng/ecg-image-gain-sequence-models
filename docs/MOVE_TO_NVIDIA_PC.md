# Moving the project to the NVIDIA PC

Written 1 Nov 2026. The MacBook Air (CPU only, 30 W charger) was too slow and ran hot, so the
remaining 12–13 training runs move to a Windows PC with an NVIDIA graphics card. Everything
finished so far travels as files; nothing has to be redone.

## Two ways to move it

- **Over the home Wi-Fi (no drive needed; used on 1 Nov 2026).** The Mac and PC must be on the same network.
- **With a USB stick or external drive** of at least 16 GB, formatted exFAT or FAT32 (not "Mac OS Extended"/APFS).

Either way, the PC needs internet access the first time to install Python and PyTorch (about 3 GB).
The passport scan is never included.

## Part 1A: over Wi-Fi (Claude can do the Mac side for you)

1. Wait until no training is running: `python -m src.status` says "No training is running".
2. On the Mac, pack the project into one file (about 12 GB on the Desktop, 10–20 minutes). The
   tool then re-reads the file to check it:

   ```bash
   cd ~/Downloads/"ecg swiss"
   /opt/anaconda3/bin/python -m src.transfer archive ~/Desktop/ecg_swiss_transfer.tar
   ```

3. Offer the file to the PC:

   ```bash
   /opt/anaconda3/bin/python -m src.transfer serve ~/Desktop/ecg_swiss_transfer.tar
   ```

   It prints an address like `http://192.168.68.114:8000/<random>/ecg_swiss_transfer.tar`. If the
   Mac asks whether Python may accept incoming connections, click **Allow**. Keep the Mac awake and
   on the charger.
4. On the PC, type that exact address into Chrome or Edge. The browser may warn that the download
   is "not secure" because it is plain http on your own network; choose **Keep**. A transfer at
   Wi-Fi speed takes roughly 5–20 minutes.
5. On the PC, open Command Prompt and unpack the file (Windows 10/11 include `tar`):

   ```bat
   cd %USERPROFILE%
   tar -xf Downloads\ecg_swiss_transfer.tar
   ```

   This creates `C:\Users\<you>\ecg swiss`. Continue with Part 2 from step 2. The setup check
   confirms that every file arrived intact.
6. Back on the Mac, press Ctrl+C in the window from step 3 to stop offering the file. Delete
   `~/Desktop/ecg_swiss_transfer.tar` once the PC check says READY.

## Part 1B: with a drive

1. Wait until no training is running, then plug in the drive (it appears as `/Volumes/<name>`).
2. In Terminal:

   ```bash
   cd ~/Downloads/"ecg swiss"
   /opt/anaconda3/bin/python -m src.transfer copy /Volumes/USB
   ```

   This copies the project to `/Volumes/USB/ecg swiss` and re-reads every file on the drive to
   check it. If the copy stops halfway, run the same command again. Add `--without-raw` to leave
   out the 3.2 GB PTB-XL download.
3. Eject the drive.

## Part 2: on the PC

Easiest: install the Claude desktop app on the PC, open the copied folder in Claude Code and
paste this message: *"This project moved from my Mac. Follow docs/MOVE_TO_NVIDIA_PC.md part 2,
then start the training queue."* To do it yourself instead:

1. **Drive only:** copy the folder to the PC's own disk, for example to `C:\Users\<you>\ecg swiss`.
   Drag it from the drive in File Explorer. Don't train from the USB drive; it is too slow.
2. **Update the NVIDIA driver** using the NVIDIA App or GeForce Experience, or from nvidia.com.
   It needs to be version 570 or newer.
3. **Install Python 3.12.10** from python.org (Windows installer, 64-bit). Tick "Add python.exe to
   PATH" during setup.
4. **Open Command Prompt** (not PowerShell) and run these one at a time:

   ```bat
   cd "C:\Users\<you>\ecg swiss"
   py -3.12 -m venv .venv
   .venv\Scripts\python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
   .venv\Scripts\python -m pip install -r requirements.txt
   ```

5. **Check everything** (takes a few minutes):

   ```bat
   .venv\Scripts\python -X utf8 -m src.check_pc_setup
   ```

   This confirms that all copied files are intact, the versions are right, the GPU is visible and
   the data opens. It also confirms that a saved model gives the same answers as it did on the Mac.
   It must end with **READY to train**. If it doesn't, send the output to Claude.
6. **Stop the PC from sleeping**: Settings → System → Power → "When plugged in, put my device to
   sleep after" → **Never**.
7. **Start the remaining runs** (one command, one line):

   ```bat
   .venv\Scripts\python -X utf8 -m src.run_queue --device cuda cnn_pool:42 cnn_pool:43 cnn_pool:44 hybrid_vit:43 hybrid_vit:44 temporal_conv:43 temporal_conv:44 cnn_lstm:43 cnn_lstm:44 fg_cnn_pool:42 fg_cnn_lstm:42 fg_temporal_conv:42 fg_hybrid_vit:42
   ```

   Leave the window open; closing it stops training. Runs that already finished are skipped
   automatically. To restart after an interruption, run the same command again. A half-finished
   run must first be moved aside, as in `experiments/abandoned_runs/`; ask Claude.

8. **Check progress** at any time from a second Command Prompt window:

   ```bat
   cd "C:\Users\<you>\ecg swiss"
   .venv\Scripts\python -m src.status
   ```

## What changed in the code for the move (for the record)

- `src/portable_cache.py`: the v1 image cache recorded the Mac's folder path and Python build.
  This loader runs every other check unchanged, including the per-image hashes and the
  numpy/OpenCV versions, but accepts a new folder and Python build. Each run records which check
  applied (`cache_binding_check` in `provenance.json`). The hash-bound
  `src/ptbxl_learning_data.py` was not edited.
- `--device cuda` for `src.run_ptbxl_experiment` and `src.run_queue`. The device is stored in each
  run's hyperparameters. GPU runs use deterministic cuBLAS and full float32 (TF32 off).
- Windows support: the experiment-log lock uses `msvcrt`, Python runs in UTF-8 mode, and log
  paths are written with `/`.
- New tools: `src.transfer` (drive copy, or one-file archive served over the local network, both with a checksum manifest), `src.check_pc_setup` (readiness and
  hardware-drift report saved to `experiments/setup_checks/`) and `src.status` (cross-platform
  progress).
