"""Run the networks of the 11 Nov 2026 amendment (40-epoch sensitivity, ResNet-18) for one or more seeds.

Usage:
    python -X utf8 -m src.run_ptbxl_extra --config experiments/configs/ptbxl_v2_long40_cnn_lstm.json --seeds 42 43 44 --device cuda

Every step is src/run_ptbxl_experiment.run, unchanged: validation-only selection, checkpoint freeze and reload
check, one test inference, patient-cluster bootstrap, append-only log. While running, it registers the
capacity-matched head and ResNet-18 in the shared loop's model table and adds their files to the hashed source
snapshot; both are restored afterwards. Run IDs are <protocol>_seed<seed>. Seeds already present are skipped,
and each seed writes experiments/logs/<run_id>.log.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
import traceback

from src import run_ptbxl_experiment as experiment
from src.common import ROOT, read_json
from src.models.ptbxl_cnn_pool_wide import LeadImageCNNPoolWide
from src.models.ptbxl_resnet18 import LeadImageResNet18
from src.training import ptbxl_trainer as trainer

EXTRA_MODELS = {"cnn_pool_wide": LeadImageCNNPoolWide, "resnet18": LeadImageResNet18}
EXTRA_CODE = ["src/run_ptbxl_extra.py", "src/models/ptbxl_cnn_pool_wide.py", "src/models/ptbxl_resnet18.py"]


@contextmanager
def extra_models():
    """Register the extra models and snapshot their code; restore the shared tables afterwards."""
    previous = list(experiment.CODE_FILES)
    trainer.MODELS.update(EXTRA_MODELS)
    experiment.CODE_FILES[:] = previous + EXTRA_CODE
    try:
        yield
    finally:
        experiment.CODE_FILES[:] = previous
        for name in EXTRA_MODELS:
            trainer.MODELS.pop(name, None)


def main(config, seeds, device):
    protocol = read_json(config)["protocol"]
    for seed in seeds:
        run_id = f"{protocol}_seed{seed}"
        if any((ROOT / "results").glob(f"*/{run_id}")):
            print(f"skip {run_id}: already present", flush=True)
            continue
        with (ROOT / "experiments/logs" / f"{run_id}.log").open("w", encoding="utf-8") as log:
            try:
                with redirect_stdout(log), redirect_stderr(log), extra_models():
                    experiment.run(config, seed, run_id, device)
                code = 0
            except Exception:
                log.write(traceback.format_exc())
                code = 1
        print(f"{run_id} exit code {code}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    arguments = parser.parse_args()
    main(arguments.config, arguments.seeds, arguments.device)
