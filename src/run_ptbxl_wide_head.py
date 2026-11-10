"""Run the capacity-matched CNN-only head (protocol ptbxl_v2_cnn_pool_wide) for one or more seeds.

Usage:
    python -X utf8 -m src.run_ptbxl_wide_head --device cuda --seeds 42 43 44 45 46

Every step is src/run_ptbxl_experiment.run, unchanged: validation-only selection, checkpoint freeze and
reload check, one test inference, patient-cluster bootstrap, append-only log. During the run,
wide_head_trainer() registers LeadImageCNNPoolWide in the shared loop's model table and adds this file
and the model file to the hashed source snapshot; both are restored afterwards.
Seeds already present are skipped; each seed writes experiments/logs/<run_id>.log.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import traceback

from src import run_ptbxl_experiment as experiment
from src.common import ROOT
from src.models.ptbxl_cnn_pool_wide import LeadImageCNNPoolWide
from src.training import ptbxl_trainer as trainer

CONFIG = ROOT / "experiments/configs/ptbxl_v2_cnn_pool_wide.json"
EXTRA_CODE = ["src/run_ptbxl_wide_head.py", "src/models/ptbxl_cnn_pool_wide.py"]


@contextmanager
def wide_head_trainer():
    """Register the wide head and snapshot its code; restore the shared tables afterwards."""
    previous = list(experiment.CODE_FILES)
    trainer.MODELS["cnn_pool_wide"] = LeadImageCNNPoolWide
    experiment.CODE_FILES[:] = previous + EXTRA_CODE
    try:
        yield
    finally:
        experiment.CODE_FILES[:] = previous
        trainer.MODELS.pop("cnn_pool_wide", None)


def main(seeds, device):
    for seed in seeds:
        run_id = f"ptbxl_v2_cnn_pool_wide_seed{seed}"
        if any((ROOT / "results").glob(f"*/{run_id}")):
            print(f"skip {run_id}: already present", flush=True)
            continue
        with (ROOT / "experiments/logs" / f"{run_id}.log").open("w", encoding="utf-8") as log:
            try:
                with redirect_stdout(log), redirect_stderr(log), wide_head_trainer():
                    experiment.run(CONFIG, seed, run_id, device)
                code = 0
            except Exception:
                log.write(traceback.format_exc())
                code = 1
        print(f"{run_id} exit code {code}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    arguments = parser.parse_args()
    main(arguments.seeds, arguments.device)
