"""Synthetic publication aggregation checks; never train or read real ECG data."""
import csv
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from src.publication_summary import AUDIT, LEGACY_FIELDS, build


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    return list(csv.DictReader(io.StringIO(path.read_text())))


def multiclass_metrics():
    names = ["Normal", "MI", "Abnormal", "History_MI"]
    matrix = [[1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    precision = [.5, 0., 1., 1.]
    recall = [1., 0., 1., 1.]
    f1 = [2/3, 0., 1., 1.]
    return dict(n_samples=4, class_names=names, confusion_matrix=matrix,
                accuracy=.75, balanced_accuracy=.75, macro_f1=sum(f1)/4, weighted_f1=sum(f1)/4,
                roc_auc_ovr_macro=.8123456789012345,
                roc_auc_per_class={name: .8123456789012345 for name in names},
                per_class={name: dict(precision=p, recall=r, **{"f1-score": f}, support=1.)
                           for name, p, r, f in zip(names, precision, recall, f1)})


def multilabel_metrics():
    names = ["NORM", "MI", "STTC", "CD", "HYP"]
    return dict(n_samples=4, n_labels=5, class_names=names, threshold=.5,
                subset_accuracy=.25, label_accuracy=.75, macro_f1=2/3, weighted_f1=2/3,
                macro_roc_auc=.8123456789012345, macro_average_precision=.7,
                per_label={name: dict(tp=1, fp=0, tn=2, fn=1, support=2, negative_support=2,
                                     precision=1., recall=.5, specificity=1., balanced_accuracy=.75,
                                     f1=2/3, roc_auc=.8123456789012345, average_precision=.7)
                           for name in names})


class Fixture:
    def __init__(self, root):
        self.root = root
        self.entries = []
        self.audit = dict(failures=[], runs={})
        self.delivery = dict(files={})

    def add(self, run_id, multilabel=False, status="completed", verified=True):
        family = "ptbxl_hybrid_vit" if multilabel else "lda"
        directory = self.root / "results" / family / run_id
        model_dir = self.root / "models" / family / run_id
        directory.mkdir(parents=True)
        model_dir.mkdir(parents=True)
        checkpoint = model_dir / ("best.pt" if multilabel else "pipeline.joblib")
        checkpoint.write_bytes(b"synthetic weights, not deserializable")
        metric = multilabel_metrics() if multilabel else multiclass_metrics()
        config = dict(seed=42, preprocessing={"synthetic": True}, class_names=metric["class_names"])
        provenance = dict(source_sha256={"src/synthetic.py": "0"*64})
        run = dict(experiment_id=run_id, status=status, model="ptbxl_image_hybrid_vit" if multilabel else "lda",
                   dataset_id="ptbxl_images_v1" if multilabel else "image_dedup_fixture",
                   split_version="synthetic_split", started_utc="2026-11-01T00:00:00+00:00",
                   completed_utc="2026-11-01T00:00:03+00:00", total_runtime_seconds=2.5,
                   model_directory=model_dir.relative_to(self.root).as_posix(), checkpoint_sha256=sha(checkpoint),
                   training=dict(model="Synthetic", config={"seed": 42}, num_parameters=123,
                                 epochs_run=2, best_epoch=1, training_seconds=1.5, runtime_seconds=2.0),
                   metrics={p: metric for p in ("train", "validation", "test")})
        write_json(directory / "run.json", run)
        write_json(directory / "config.json", config)
        write_json(directory / "provenance.json", provenance)
        for partition in ("train", "validation", "test"):
            write_json(directory / f"{partition}_metrics.json", metric)
            (directory / f"{partition}_predictions.csv").write_text("synthetic\n1\n")
        hashes = {p.name: sha(p) for p in directory.iterdir() if p.is_file()}
        if verified:
            if multilabel:
                write_json(directory / "verification.json", dict(status="passed", run_id=run_id, failures=[],
                           checkpoint_sha256=sha(checkpoint), run_artifact_sha256=hashes))
            else:
                self.audit["runs"][run_id] = dict(status="completed", run_sha256=sha(directory / "run.json"),
                    **{f"{p}_metrics": metric for p in ("train", "validation", "test")},
                    fresh_verification={"partitions": {p: {"all_saved_metrics_recomputed": True} for p in ("train", "validation", "test")}},
                    model_artifact_inventory={checkpoint.name: {"sha256": sha(checkpoint)}})
                for name, digest in hashes.items():
                    self.delivery["files"][(directory / name).relative_to(self.root).as_posix()] = {"sha256": digest}
        self.entries.append(dict(experiment_id=run_id, model=family, status=status,
                            dataset_version=run["dataset_id"], split_version="synthetic_split",
                            output_directory=directory.relative_to(self.root).as_posix()))
        return directory

    def finish(self):
        path = self.root / "experiments/experiment_log.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=LEGACY_FIELDS)
            writer.writeheader()
            writer.writerows(self.entries)
        write_json(self.root / AUDIT, self.audit)
        write_json(self.root / "experiments/logs/delivery_manifest.json", self.delivery)


class PublicationSummaryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.fixture = Fixture(self.root)
        self.fixture.add("lda_v1_seed42")
        self.hybrid = self.fixture.add("ptbxl_hybrid_vit_image_v1_seed42", multilabel=True)
        self.fixture.finish()
        self.output = self.root / "new_review"
        self.master = self.output / "master.csv"

    def tearDown(self):
        self.temp.cleanup()

    def run_build(self):
        return build(self.root, self.output, self.master)

    def test_cohort_definitions_precision_and_rerun_preservation(self):
        before = {str(p): sha(p) for p in self.root.rglob("*") if p.is_file()}
        result = self.run_build()
        rows = read_csv(self.master)
        self.assertEqual(len(rows), 2)
        lda, hybrid = rows
        self.assertEqual(lda["parameters"], "NA")
        self.assertEqual(hybrid["parameters"], "123")
        self.assertEqual(lda["git_commit"], "unavailable")
        self.assertEqual(float(lda["test_accuracy"]), .75)
        self.assertEqual(float(hybrid["test_accuracy"]), .25)
        self.assertEqual(float(hybrid["test_label_accuracy"]), .75)
        self.assertEqual(float(hybrid["test_balanced_accuracy"]), .75)
        self.assertEqual(float(hybrid["test_macro_recall"]), .5)
        self.assertEqual(float(hybrid["test_macro_roc_auc"]), .8123456789012345)
        self.assertEqual(float(hybrid["utc_wall_span_seconds"]), 3.)
        self.assertEqual(float(hybrid["runner_elapsed_seconds"]), 2.5)
        comparison = read_csv(self.output / "publication_model_comparison.csv")
        labels = read_csv(self.output / "publication_model_per_label_metrics.csv")
        self.assertEqual(len(comparison), 6)
        self.assertEqual(len(labels), 27)
        self.assertIn("Mendeley four-class", (self.output / "publication_model_comparison.md").read_text())
        self.assertFalse(result["ranking_between_cohorts"])
        for path, digest in result["output_artifact_sha256"].items():
            self.assertEqual(sha(Path(path)), digest)
        first_csv = self.master.read_bytes()
        self.run_build()
        self.assertEqual(first_csv, self.master.read_bytes())
        for path, digest in before.items():
            self.assertEqual(sha(Path(path)), digest)

    def test_failed_cancelled_and_unverified_have_no_promoted_metrics(self):
        for name, status in [("failed_run", "failed"), ("cancelled_run", "cancelled"),
                             ("pending_run", "running"), ("unverified_run", "completed")]:
            self.fixture.add(name, multilabel=True, status=status, verified=False)
        self.fixture.finish()
        report = self.run_build()
        self.assertEqual(report["run_count"], 6)
        self.assertEqual(len(report["excluded_runs"]), 4)
        rows = read_csv(self.master)
        for row in rows[2:]:
            self.assertEqual(row["test_accuracy"], "")
            self.assertEqual(row["verification_status"], "not_verified")
            self.assertTrue(row["exclusion_reason"])
        self.assertEqual([r["status"] for r in rows[2:]], ["failed", "cancelled", "running", "completed"])

    def test_failed_central_entry_without_artifacts_is_retained(self):
        self.fixture.entries.append(dict(experiment_id="cancelled_before_start", status="cancelled", model="future"))
        self.fixture.finish()
        report = self.run_build()
        self.assertEqual(report["excluded_runs"][0]["status"], "cancelled")
        self.assertEqual(report["run_count"], 3)

    def test_completed_unlogged_run_is_discovered(self):
        self.fixture.entries.pop()
        self.fixture.finish()
        report = self.run_build()
        self.assertEqual(report["run_count"], 2)
        self.assertEqual(len(report["verified_completed_runs"]), 2)

    def test_runner_attempt_without_terminal_run_is_included_from_saved_progress(self):
        directory = self.fixture.add("active_attempt", multilabel=True, verified=False)
        # Real runner shape: provenance/config/progress exist, with no run.json
        # or central row until terminal completion/failure.
        (directory / "run.json").unlink()
        self.fixture.entries.pop()
        config_path = directory / "config.json"
        config = json.loads(config_path.read_text())
        config.update(model="ptbxl_image_hybrid_vit", dataset_id="ptbxl_images_v1",
                      split_version="synthetic_split", training={"seed": 42})
        write_json(config_path, config)
        write_json(directory / "provenance.json", {"started_utc": "2020-01-01T00:00:00+00:00"})
        progress_path = directory / "evaluation_progress.json"
        write_json(progress_path, {"status": "running", "phase": "training",
                                   "updated_utc": "2020-01-01T00:00:03+00:00"})
        self.fixture.finish()
        report = self.run_build()
        row = read_csv(self.master)[-1]
        self.assertEqual(report["run_count"], 3)
        self.assertEqual(row["experiment_id"], "active_attempt")
        self.assertEqual(row["status"], "running")
        self.assertEqual(row["central_log_status"], "")
        self.assertEqual(row["status_last_updated_utc"], "2020-01-01T00:00:03+00:00")
        self.assertEqual(row["liveness_verified"], "False")
        self.assertEqual(row["verification_status"], "not_verified")
        self.assertEqual(row["test_accuracy"], "")
        self.assertEqual(row["parameters"], "unavailable")
        self.assertIn("process liveness has not been verified", row["limitations"])
        self.assertIn(progress_path.relative_to(self.root).as_posix(), report["source_artifact_sha256"])
        self.assertEqual(len(read_csv(self.output / "publication_model_comparison.csv")), 6)
        # An old timestamp is not evidence of death, and no progress is not
        # evidence of either completion or failure.
        progress_path.unlink()
        self.run_build()
        row = read_csv(self.master)[-1]
        self.assertEqual(row["status"], "incomplete")
        self.assertEqual(row["status_last_updated_utc"], "unavailable")
        self.assertEqual(row["test_accuracy"], "")

    def test_tampered_artifacts_fail_before_writes(self):
        (self.hybrid / "test_metrics.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            self.run_build()
        self.assertFalse(self.output.exists())

    def test_manifest_cannot_escape_run_directory(self):
        path = self.hybrid / "verification.json"
        content = json.loads(path.read_text())
        content["run_artifact_sha256"]["../../../experiments/experiment_log.csv"] = sha(self.root / "experiments/experiment_log.csv")
        write_json(path, content)
        with self.assertRaisesRegex(ValueError, "escapes run directory"):
            self.run_build()

    def test_legacy_and_immutable_output_guards(self):
        legacy = self.root / "experiments/experiment_log.csv"
        before = sha(legacy)
        with self.assertRaisesRegex(ValueError, "legacy log"):
            build(self.root, self.output, legacy)
        with self.assertRaisesRegex(ValueError, "immutable"):
            build(self.root, self.hybrid, self.master)
        self.assertEqual(sha(legacy), before)

    def test_failed_certificate_never_promotes_completed_run(self):
        path = self.hybrid / "verification.json"
        content = json.loads(path.read_text())
        content["failures"] = ["fixture failure"]
        write_json(path, content)
        report = self.run_build()
        self.assertEqual(len(report["verified_completed_runs"]), 1)
        self.assertEqual(report["excluded_runs"][0]["reason"], "per-run verification has not passed")

    def test_verification_must_bind_predictions(self):
        path = self.hybrid / "verification.json"
        content = json.loads(path.read_text())
        del content["run_artifact_sha256"]["test_predictions.csv"]
        write_json(path, content)
        with self.assertRaisesRegex(ValueError, "does not bind required"):
            self.run_build()

    def test_central_schema_is_not_silently_reinterpreted(self):
        (self.root / "experiments/experiment_log.csv").write_text("experiment_id,status\nx,failed\n")
        with self.assertRaisesRegex(ValueError, "schema changed"):
            self.run_build()

    def test_unrelated_output_file_and_symlink_are_not_overwritten(self):
        unrelated = self.root / "original.md"
        unrelated.write_text("preserve original")
        with self.assertRaisesRegex(ValueError, "unrelated existing"):
            build(self.root, self.output, unrelated)
        self.output.mkdir()
        target = self.output / "publication_model_comparison.csv"
        try:
            target.symlink_to(self.hybrid / "run.json")
        except OSError:  # Windows without Developer Mode cannot create symlinks
            self.skipTest("symlinks are not permitted for this user")
        before = sha(self.hybrid / "run.json")
        with self.assertRaisesRegex(ValueError, "immutable"):
            self.run_build()
        self.assertEqual(sha(self.hybrid / "run.json"), before)
        self.assertEqual(unrelated.read_text(), "preserve original")


if __name__ == "__main__":
    unittest.main()
