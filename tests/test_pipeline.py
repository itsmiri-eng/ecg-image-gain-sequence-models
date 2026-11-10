"""Synthetic checks for split isolation, frozen evidence, PCA fitting, and metrics."""

import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
from PIL import Image

from src.common import ROOT, read_json, sha256_file, write_csv
from src.evaluation.metrics import evaluate, save_evaluation
from src.models.lda import train_lda
from src.prepare import prepare
from src.verify import load_verified_data


CLASS_NAMES = ["Normal", "MI", "Abnormal", "History_MI"]


class EvaluationTest(unittest.TestCase):
    def test_metrics_match_hand_calculated_imbalanced_confusion_matrix(self):
        y = np.repeat(np.arange(4), [4, 3, 2, 1])
        pred = np.array([0, 0, 1, 1, 1, 1, 2, 2, 2, 0])
        probabilities = np.full((len(y), 4), 0.1)
        probabilities[np.arange(len(y)), pred] = 0.7
        result = evaluate(y, probabilities, CLASS_NAMES)
        self.assertEqual(result["confusion_matrix"], [[2, 2, 0, 0], [0, 2, 1, 0], [0, 0, 2, 0], [1, 0, 0, 0]])
        self.assertAlmostEqual(result["accuracy"], 0.6)
        self.assertAlmostEqual(result["balanced_accuracy"], 13 / 24)
        self.assertAlmostEqual(result["macro_f1"], 17 / 35)
        self.assertAlmostEqual(result["weighted_f1"], 0.56)
        self.assertAlmostEqual(result["per_class"]["Normal"]["precision"], 2 / 3)
        self.assertAlmostEqual(result["per_class"]["MI"]["recall"], 2 / 3)
        self.assertEqual(result["per_class"]["History_MI"]["f1-score"], 0.0)
        # Independently computed positive-negative ranking probabilities, ties=0.5.
        class_aucs = [2 / 3, 29 / 42, 15 / 16, 1 / 2]
        for name, expected in zip(CLASS_NAMES, class_aucs):
            self.assertAlmostEqual(result["roc_auc_per_class"][name], expected)
        self.assertAlmostEqual(result["roc_auc_ovr_macro"], np.mean(class_aucs))
        self.assertAlmostEqual(result["roc_auc_ovr_weighted"], np.average(class_aucs, weights=[4, 3, 2, 1]))
        with tempfile.TemporaryDirectory() as directory:
            rows = [{"sample_id": f"sample-{i}", "path": f"source-{i}.png", "label": int(y[i])} for i in range(len(y))]
            save_evaluation(directory, "test", rows, y, probabilities, CLASS_NAMES)
            with (Path(directory) / "test_predictions.csv").open() as handle:
                predictions = list(csv.DictReader(handle))
            self.assertEqual(len(predictions), len(y))
            self.assertEqual([r["sample_id"] for r in predictions], [r["sample_id"] for r in rows])
            self.assertTrue((Path(directory) / "test_confusion_matrix.png").exists())

    def test_invalid_probabilities_fail_and_absent_class_auc_is_not_invented(self):
        for invalid in (
            np.full((4, 4), np.nan),
            np.full((4, 4), 0.5),
            np.array([[-0.1, 0.3, 0.4, 0.4]] * 4),
            np.full((4, 3), 1 / 3),
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    evaluate(np.arange(4), invalid, CLASS_NAMES)
        result = evaluate(np.array([0, 1, 2]), np.eye(4)[:3], CLASS_NAMES)
        self.assertIsNone(result["roc_auc_ovr_macro"])


class LDATrainOnlyTest(unittest.TestCase):
    def test_pca_centers_on_training_only_and_checkpoint_round_trip(self):
        rng = np.random.default_rng(81)
        x_train = rng.uniform(0, 0.3, size=(24, 1, 4, 8)).astype(np.float32)
        y_train = np.tile(np.arange(4), 6)
        heldout = rng.uniform(0.8, 1.0, size=(8, 1, 4, 8)).astype(np.float32)
        config = {"pca_components": 6, "pca_solver": "randomized", "solver": "lsqr", "shrinkage": "auto", "priors": "uniform"}
        with tempfile.TemporaryDirectory() as directory:
            predictor, metadata = train_lda(x_train, y_train, config, 42, directory)
            np.testing.assert_allclose(
                predictor.pipeline["pca"].mean_, x_train.reshape(24, -1).mean(axis=0), atol=1e-7
            )
            saved_mean = predictor.pipeline["pca"].mean_.copy()
            probabilities = predictor.predict_proba(heldout)
            self.assertEqual(probabilities.shape, (8, 4))
            np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
            np.testing.assert_array_equal(saved_mean, predictor.pipeline["pca"].mean_)
            restored = joblib.load(Path(directory) / "pipeline.joblib")
            np.testing.assert_allclose(probabilities, restored.predict_proba(heldout.reshape(8, -1)), atol=1e-6)
            self.assertEqual(metadata["pca_components"], 6)
            np.testing.assert_array_equal(restored["lda"].classes_, np.arange(4))
            with self.assertRaises(FileExistsError):
                train_lda(x_train, y_train, config, 42, directory)


class FrozenSplitTest(unittest.TestCase):
    """Exercise preparation and verification against a wholly synthetic cohort."""

    def test_duplicates_report_ids_and_audited_links_cannot_cross_splits(self):
        config = read_json(ROOT / "experiments/configs/baseline_v1.json")
        rng = np.random.default_rng(170)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data/processed").mkdir(parents=True)
            original_rows, paths, source_hashes = [], {}, {}
            source_pixels = None
            for label, folder in enumerate(config["class_folders"]):
                image_dir = root / config["raw_directory"] / folder
                image_dir.mkdir(parents=True)
                for sample in range(20):
                    pixels = rng.integers(0, 256, size=(40, 80, 3), dtype=np.uint8)
                    path = image_dir / f"sample_{sample:02d}.png"
                    Image.fromarray(pixels).save(path, compress_level=0)
                    file_hash = sha256_file(path)
                    paths[label, sample] = path
                    source_hashes[label, sample] = file_hash
                    # Repeat a report ID across classes: conservative identity union.
                    report_id = "shared-report" if label in (0, 1) and sample == 0 else f"report-{label}-{sample}"
                    original_rows.append({"file_sha256": file_hash, "printed_id": report_id, "status": "synthetic"})
                    if label == 0 and sample == 0:
                        source_pixels = pixels
            duplicate = paths[0, 0].parent / "zz_recompressed_exact_duplicate.png"
            Image.fromarray(source_pixels).save(duplicate, compress_level=9)
            duplicate_hash = sha256_file(duplicate)
            self.assertNotEqual(duplicate_hash, source_hashes[0, 0])
            original_rows.append({"file_sha256": duplicate_hash, "printed_id": "shared-report", "status": "synthetic"})
            write_csv(root / "data/audit/header_ids.csv", original_rows)
            # Link references an excluded copy, so it must map to its retained pixel group.
            write_csv(root / "data/audit/near_duplicate_links.csv", [{
                "sha256_a": duplicate_hash, "sha256_b": source_hashes[2, 0], "reason": "synthetic test link"
            }])
            config_path = root / "baseline.json"
            config_path.write_text(json.dumps(config))
            with patch("src.prepare.ROOT", root), patch("src.verify.ROOT", root):
                with contextlib.redirect_stdout(io.StringIO()):
                    prepare(config_path)
                x, y, rows, partitions, protocol = load_verified_data()
                self.assertEqual(x.shape, (80, 1, 128, 256))
                self.assertEqual(len(y), 80)
                self.assertEqual(protocol["audit"]["source_samples"], 81)
                self.assertEqual(protocol["audit"]["redundant_exact_copies_excluded"], 1)
                by_hash = {row["file_sha256"]: row for row in rows}
                identity_group = by_hash[source_hashes[0, 0]]["group_id"]
                self.assertEqual(identity_group, by_hash[source_hashes[1, 0]]["group_id"])
                self.assertEqual(identity_group, by_hash[source_hashes[2, 0]]["group_id"])
                self.assertEqual(sum(map(len, partitions.values())), 80)
                for split in partitions.values():
                    self.assertEqual(set(y[split]), {0, 1, 2, 3})
                split_before = sha256_file(root / "data/splits/fixed_split.csv")
                with self.assertRaises(FileExistsError):
                    prepare(config_path)
                self.assertEqual(split_before, sha256_file(root / "data/splits/fixed_split.csv"))
                for relative_path in (
                    "data/splits/fixed_split.csv",
                    "data/audit/header_ids.csv",
                    "data/audit/near_duplicate_links.csv",
                ):
                    audit_path = root / relative_path
                    original_bytes = audit_path.read_bytes()
                    audit_path.write_bytes(original_bytes + b"tampered")
                    with self.assertRaisesRegex(ValueError, "changed"):
                        load_verified_data()
                    audit_path.write_bytes(original_bytes)
                protocol_path = root / "data/splits/protocol.json"
                frozen_protocol_bytes = protocol_path.read_bytes()
                altered_protocol = json.loads(frozen_protocol_bytes)
                altered_protocol["config"]["seed"] += 1
                protocol_path.write_text(json.dumps(altered_protocol))
                with self.assertRaisesRegex(ValueError, "configuration changed"):
                    load_verified_data()
                protocol_path.write_bytes(frozen_protocol_bytes)
                # Byte changes to either frozen input or source evidence must be rejected.
                cache_path = root / "data/processed/inputs_v1.npz"
                cache_bytes = cache_path.read_bytes()
                cache_path.write_bytes(cache_bytes + b"tampered")
                with self.assertRaises((AssertionError, ValueError)):
                    load_verified_data()
                cache_path.write_bytes(cache_bytes)
                changed_original = paths[3, 1]
                changed_original.write_bytes(changed_original.read_bytes() + b"tampered")
                with self.assertRaises((AssertionError, ValueError)):
                    load_verified_data()


if __name__ == "__main__":
    unittest.main()
