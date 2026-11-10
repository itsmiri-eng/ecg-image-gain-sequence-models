"""Run a list of PTB-XL v2 experiments sequentially, skipping ones already present.

Usage (two lanes were used in November 2026, each with 4 CPU threads):
    python -m src.run_queue cnn_pool:42 hybrid_vit:43 temporal_conv:43 cnn_lstm:43 cnn_pool:44
    python -m src.run_queue cnn_pool:43 hybrid_vit:44 temporal_conv:44 cnn_lstm:44

Each item is <name>:<seed> and uses experiments/configs/ptbxl_v2_<name>.json (e.g. fg_hybrid_vit:42
for the fixed-gain condition, config ptbxl_v2_fg_hybrid_vit.json).
A failed run is recorded (failure.json, log row) and the queue continues.
Runtimes of concurrently executed runs are not controlled speed measurements.
`--device cuda` (first argument) trains on an NVIDIA GPU; the device is recorded in each run.
"""
import subprocess
import sys

from src.common import ROOT

device = []
items = sys.argv[1:]
if items[:1] == ["--device"]:
    device, items = items[:2], items[2:]
for item in items:
    model, seed = item.split(":")
    run_id = f"ptbxl_v2_{model}_seed{seed}"
    if any((ROOT / "results").glob(f"*/{run_id}")):
        print(f"skip {run_id}: already present", flush=True)
        continue
    with (ROOT / "experiments/logs" / f"{run_id}.log").open("w") as log:
        code = subprocess.call([sys.executable, "-X", "utf8", "-u", "-m", "src.run_ptbxl_experiment",
                                "--config", f"experiments/configs/ptbxl_v2_{model}.json", "--seed", seed, *device],
                               cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    print(f"{run_id} exit code {code}", flush=True)
