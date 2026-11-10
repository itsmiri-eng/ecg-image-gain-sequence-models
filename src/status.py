"""Plain-language progress of the PTB-XL training runs (macOS, Linux and Windows).

Usage:
    python -m src.status

Reads only the files the runs write (progress.json, history.csv, run.json, failure.json).
PLANNED is the remaining queue moved to the NVIDIA PC on 1 Nov 2026 (docs/MOVE_TO_NVIDIA_PC.md).
"""
import csv
import json

from src.common import ROOT

PLANNED = ["cnn_pool:42", "cnn_pool:43", "cnn_pool:44", "hybrid_vit:43", "hybrid_vit:44", "temporal_conv:43",
           "temporal_conv:44", "cnn_lstm:43", "cnn_lstm:44", "fg_cnn_pool:42", "fg_cnn_lstm:42",
           "fg_temporal_conv:42", "fg_hybrid_vit:42"]


def main():
    running = sorted(ROOT.glob("models/*/*/.training.lock"))
    print("Training is RUNNING." if running else "No training is running right now.")
    for lock in running:
        progress = json.loads((lock.parent / "progress.json").read_text()) if (lock.parent / "progress.json").exists() else {}
        history = lock.parent / "history.csv"
        rows = list(csv.DictReader(history.open())) if history.exists() else []
        best = max((float(r["val_macro_roc_auc"]) for r in rows), default=None)
        print(f"  {lock.parent.name}: epoch {progress.get('epoch', 0)} of {progress.get('max_epochs', '?')}, "
              f"batch {progress.get('batch', 0)} of {progress.get('batches', '?')}"
              + (f", best validation AUROC so far {best:.4f}" if best is not None else ""))
    done, failed, waiting = [], [], []
    for item in PLANNED:
        model, seed = item.split(":")
        run_id = f"ptbxl_v2_{model}_seed{seed}"
        if any(ROOT.glob(f"results/*/{run_id}/run.json")):
            done.append(run_id)
        elif any(ROOT.glob(f"results/*/{run_id}/failure.json")):
            failed.append(run_id)
        elif not any(lock.parent.name == run_id for lock in running):
            waiting.append(run_id)
    print(f"\nPlanned runs finished: {len(done)} of {len(PLANNED)}")
    for run_id in done:
        print(f"  done: {run_id}")
    for run_id in failed:
        print(f"  FAILED: {run_id}  <- tell Claude")
    if waiting:
        print(f"Not started yet: {len(waiting)}")


if __name__ == "__main__":
    main()
