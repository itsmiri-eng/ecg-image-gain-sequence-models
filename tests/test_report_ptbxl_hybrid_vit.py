"""Synthetic saved-prediction fixtures; never construct or run an ECG model."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src import report_ptbxl_hybrid_vit as report
from src.evaluation.multilabel import evaluate_multilabel
from src.evaluation.ptbxl_uncertainty import patient_cluster_bootstrap


Y = np.asarray([[1, 0, 1, 0, 1], [0, 1, 0, 1, 0]] * 2)
P = np.where(Y, .8, .2)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def refresh_verification(out, run_id):
    run = json.loads((out / "run.json").read_text())
    provenance = json.loads((out / "provenance.json").read_text())
    write_json(out / "verification.json", {
        "status": "passed", "run_id": run_id, "failures": [],
        "checkpoint_sha256": run["checkpoint_sha256"], "configuration_sha256": provenance["configuration_sha256"],
        "synthetic_fixture_only": True,
        "run_artifact_sha256": {str(path.relative_to(out)): report._sha(path) for path in out.rglob("*")
                                if path.is_file() and path.name != "verification.json"},
    })


def fixture(root, family, run_id, probabilities):
    cfg = json.loads((report.ROOT / "experiments/configs" / (family + "_image_v1.json")).read_text())
    cfg.update(cache_root="data/cache", source_dataset_root="data/source")
    out, model = root / "results" / family / run_id, root / "models" / family / run_id
    out.mkdir(parents=True); model.mkdir(parents=True)
    (model / "best.pt").write_bytes(b"Synthetic test checkpoint: must never be loaded")
    checkpoint = report._sha(model / "best.pt")
    cache = root / "data/cache"
    cache.mkdir(parents=True, exist_ok=True)
    for name in ("protocol.json", "metadata.json", "cache_manifest.csv"):
        (cache / name).write_text("Synthetic shared cache binding")
    sources = {}
    for name in ("src/evaluation/multilabel.py", "src/evaluation/ptbxl_uncertainty.py"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((report.ROOT / name).read_bytes())
        sources[name] = report._sha(path)
    write_json(out / "config.json", cfg)
    config_hash = report._sha(out / "config.json")
    normalization = {"fit_partition": "train", "mean": .1, "std": .2}
    provenance = {"configuration_sha256": config_hash, "normalization": normalization,
                  "selection_rule": cfg["selection_rule"], "source_sha256": sources,
                  "cache_sha256": {name: report._sha(cache / name) for name in ("protocol.json", "metadata.json", "cache_manifest.csv")}}
    metrics, predictions = {}, {}
    for split_index, split in enumerate(report.PARTITIONS):
        rows = []
        for index, (truth, p) in enumerate(zip(Y, probabilities)):
            row = {"ecg_id": str(split_index * 10 + index + 1), "patient_id": str(split_index * 10 + index // 2 + 1),
                   "image_path": f"images/{split}/{index}.png"}
            for label, target, probability in zip(report.CLASS_NAMES, truth, p):
                row.update({f"truth_{label}": int(target), f"probability_{label}": float(probability), f"prediction_{label}": int(probability >= .5)})
            rows.append(row)
        predictions[split] = rows
        report._write_csv(out / f"{split}_predictions.csv", rows)
        metrics[split] = evaluate_multilabel(Y, probabilities, report.CLASS_NAMES)
        write_json(out / f"{split}_metrics.json", metrics[split])
    family_names = {"ptbxl_hybrid_vit": ("LeadImageHybridViT", 23881), "ptbxl_cnn_lstm": ("LeadImageCNNLSTM", 18025),
                    "ptbxl_spatiotemporal": ("LeadImageSpatiotemporal", 17161)}
    name, parameters = family_names[family]
    validation = metrics["validation"]
    key = [validation["macro_roc_auc"], validation["macro_average_precision"], -validation["binary_cross_entropy"]]
    training = {"model": name, "config": cfg["training"], "normalization": normalization, "class_names": report.CLASS_NAMES,
                "num_parameters": parameters, "runtime_seconds": 12.5, "best_epoch": 1, "epochs_run": 5,
                "best_selection_key": key, "stopped_early": True, "best_validation_metrics": validation,
                "selected_epoch_online_train_bce": .5, "device": "cpu"}
    history = []
    for epoch in range(1, 6):
        row = {"epoch": epoch, "learning_rate": .001, "train_online_bce": .5,
               **{f"val_{k}": v for k, v in validation.items() if isinstance(v, (int, float)) and not isinstance(v, bool)},
               "checkpoint_improved": str(epoch == 1), "epochs_without_selection_improvement": epoch - 1,
               "max_gradient_norm_before_clip": 1., "epoch_seconds": 2.}
        row["val_macro_roc_auc"] -= .01 * (epoch - 1)
        history.append(row)
    report._write_csv(model / "history.csv", history)
    run = {"status": "completed", "experiment_id": run_id, "model": cfg["model"], "dataset_id": cfg["dataset_id"],
           "split_version": cfg["split_version"], "threshold": .5, "training": training, "metrics": metrics,
           "checkpoint_sha256": checkpoint, "test_inference_calls": 1, "total_runtime_seconds": 20.,
           "started_utc": "2026-11-01T00:00:00+00:00", "completed_utc": "2026-11-01T00:01:00+00:00",
           "model_directory": str(model.relative_to(root))}
    frozen = {"training": training, "threshold": .5, "test_inference_started": False, "selection_rule": cfg["selection_rule"],
              "checkpoint_sha256": checkpoint, "configuration_sha256": config_hash}
    state = {"status": "completed", "inference_calls": 1, "threshold": .5, "checkpoint_sha256": checkpoint, "records": len(Y)}
    for filename, value in (("run.json", run), ("provenance.json", provenance), ("training_metadata.json", training),
                            ("selection_frozen.json", frozen), ("test_inference_state.json", state)):
        write_json(out / filename, value)
    if family == "ptbxl_hybrid_vit":
        # Both synthetic patients carry the identical two records. All patient
        # resamples therefore have identical scores; 500 equal replicates are
        # represented without repeatedly calculating the same synthetic scores.
        ci = patient_cluster_bootstrap(Y, probabilities, [r["patient_id"] for r in predictions["test"]], report.CLASS_NAMES, n_resamples=2, seed=2024)
        ci["n_resamples"] = 500
        for metric in ci["metrics"].values():
            metric["defined_resamples"] = 500
        write_json(out / "test_patient_bootstrap.json", ci)
    refresh_verification(out, run_id)
    return out


class HybridReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="synthetic-hybrid-report-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "docs").mkdir()
        (self.root / "docs/ptbxl_image_dataset.md").write_text("Current factual dataset prose; synthetic context only")
        hybrid = P.copy(); hybrid[:, 4] = 1 - hybrid[:, 4]
        lstm = P.copy(); lstm[:, 0] = 1 - lstm[:, 0]
        fixture(self.root, "ptbxl_spatiotemporal", report.REFERENCES[0][2], P)
        fixture(self.root, "ptbxl_cnn_lstm", report.REFERENCES[1][2], lstm)
        self.out = fixture(self.root, "ptbxl_hybrid_vit", report.DEFAULT_RUN, hybrid)

    def test_loader_requires_passed_hash_bound_verifications(self):
        bundle = report._load_bundle(self.root, report.DEFAULT_RUN)
        self.assertEqual(set(bundle["references"]), {"tcn_primary", "cnn_lstm_secondary"})
        self.assertEqual(bundle["hybrid"]["training"]["num_parameters"], 23881)
        with (self.out / "test_predictions.csv").open("a") as handle:
            handle.write("corrupted")
        with self.assertRaisesRegex(ValueError, "hash changed"):
            report._load_bundle(self.root, report.DEFAULT_RUN)

    def test_changed_dataset_prose_is_context_not_old_numeric_hash(self):
        (self.root / "docs/ptbxl_image_dataset.md").write_text("Updated factual provenance prose")
        bundle = report._load_bundle(self.root, report.DEFAULT_RUN)
        self.assertIn(str((self.root / "docs/ptbxl_image_dataset.md").resolve()), bundle["inputs"].hashes)

    def test_all_partitions_must_align_even_when_test_matches(self):
        path = self.out / "train_predictions.csv"
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        rows[0]["patient_id"] = "9999"
        report._write_csv(path, rows)
        refresh_verification(self.out, report.DEFAULT_RUN)
        with self.assertRaisesRegex(ValueError, "paired alignment"):
            report._load_bundle(self.root, report.DEFAULT_RUN)

    def test_class_order_changes_rejected(self):
        path = self.out / "config.json"
        cfg = json.loads(path.read_text()); cfg["class_names"] = list(reversed(cfg["class_names"]))
        write_json(path, cfg); refresh_verification(self.out, report.DEFAULT_RUN)
        with self.assertRaisesRegex(ValueError, "class order"):
            report._load_bundle(self.root, report.DEFAULT_RUN)

    def test_failed_verification_prevents_reporting_and_bootstrap(self):
        path = self.out / "verification.json"
        verification = json.loads(path.read_text()); verification["status"] = "failed"
        write_json(path, verification)
        with patch.object(report, "paired_patient_bootstrap", side_effect=AssertionError("must not resample")):
            with self.assertRaises(ValueError):
                report.generate_report(root=self.root)
        self.assertFalse((self.out / "figures").exists())

    def test_full_synthetic_report_preserves_sources_and_reports_deterioration(self):
        before = {path: report._sha(path) for path in self.root.rglob("*") if path.is_file()}
        original_bootstrap = report.paired_patient_bootstrap
        with patch.object(report, "paired_patient_bootstrap", wraps=original_bootstrap) as paired:
            produced = report.generate_report(root=self.root)
        self.assertEqual(paired.call_count, 2)
        for call in paired.call_args_list:
            self.assertEqual(call.kwargs, {"n_resamples": 500, "seed": 2024})
        for path, digest in before.items():
            self.assertEqual(report._sha(path), digest)
        comparison = Path(produced["comparison_directory"])
        obj = json.loads((comparison / "comparison.json").read_text())
        first = obj["comparisons"]["tcn_primary"]
        self.assertLess(first["paired_test_bootstrap"]["metrics"]["macro_roc_auc"]["difference"], 0)
        self.assertEqual(set(first["paired_test_bootstrap"]["metrics"]), set(report.METRICS))
        text = (comparison / "comparison.md").read_text()
        self.assertIn("hybrid test macro ROC-AUC is lower", text)
        self.assertIn("Labels with lower hybrid recall at 0.5: **HYP**", text)
        self.assertIn("not a capacity-controlled isolation", text)
        self.assertIn("not pure CPU time", text)
        main = Path(produced["report"]).read_text()
        self.assertIn("**23,881-parameter**", main)
        self.assertIn("not a vanilla raw-patch ViT", main)
        self.assertIn("no inference", main)
        self.assertIn("offline classifier", main)
        self.assertIn("Lowest test recall at 0.5: **HYP**", main)
        with (comparison / "per_label_differences.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 30)
        hyp = next(row for row in rows if row["comparison"] == "tcn_primary" and row["partition"] == "test" and row["label"] == "HYP")
        self.assertEqual(int(hyp["delta_tp"]), -2)
        self.assertEqual(int(hyp["delta_fn"]), 2)
        self.assertLess(float(hyp["delta_recall"]), 0)
        for directory, names in ((self.out / "figures", report.FIGURES), (comparison, ("comparison", "per_label_differences"))):
            for name in names:
                for extension in ("png", "svg"):
                    self.assertGreater((directory / f"{name}.{extension}").stat().st_size, 1000)
        manifest = json.loads((comparison / "manifest.json").read_text())
        for path, digest in manifest["output_sha256"].items():
            self.assertEqual(report._sha(Path(path)), digest)
        with patch.object(report, "_load_bundle", side_effect=AssertionError("should fail before reads")):
            with self.assertRaises(FileExistsError):
                report.generate_report(root=self.root)

    def test_existing_or_unrelated_outputs_are_protected(self):
        figures = self.out / "figures"
        figures.mkdir(); (figures / "unrelated.txt").write_text("preserve")
        with self.assertRaisesRegex(ValueError, "unrelated"):
            report.generate_report(root=self.root, overwrite=True)
        self.assertEqual((figures / "unrelated.txt").read_text(), "preserve")

    def test_comparison_findings_include_zero_and_undefined(self):
        pair = {"reference_title": "TCN (primary)", "paired_test_bootstrap": {"metrics": {"macro_roc_auc": {
            "difference": .01, "ci_lower": 0., "ci_upper": .02, "interpretation": "includes_zero_no_resolved_difference"}}}}
        self.assertIn("unresolved", report._finding(pair))
        pair["paired_test_bootstrap"]["metrics"]["macro_roc_auc"].update(difference=None, ci_lower=None, ci_upper=None, interpretation="not_estimable")
        self.assertIn("not estimable", report._finding(pair))


if __name__ == "__main__":
    unittest.main()
