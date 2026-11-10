"""Synthetic-only integrity and geometry checks; never prepare the real cache."""
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from src import ptbxl_learning_data as data


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value))


class LearningDataTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.counts = {"train": 2, "validation": 1, "test": 1}
        self.patch_counts = patch.object(data, "EXPECTED_COUNTS", self.counts)
        self.patch_records = patch.object(data, "EXPECTED_SOURCE_RECORDS", 4)
        self.patch_counts.start()
        self.patch_records.start()
        self.addCleanup(self.patch_counts.stop)
        self.addCleanup(self.patch_records.stop)

    def fixture_source(self):
        source = self.root / "source"
        source.mkdir()
        rows = []
        for ecg_id, fold, ink in ((1, 1, 0), (2, 2, 128), (3, 9, 255), (4, 10, 255)):
            split = "train" if fold <= 8 else "validation" if fold == 9 else "test"
            image = np.full(data.PAGE_SHAPE, 255 - ink, np.uint8)
            ok, buffer = cv2.imencode(".png", image)
            self.assertTrue(ok)
            payload = buffer.tobytes()
            relative = f"images/{split}/ecg_{ecg_id:05d}.png"
            destination = source / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
            labels = ["NORM", "MI"] if ecg_id == 1 else ["MI"]
            row = {"ecg_id": ecg_id, "patient_id": 100 + ecg_id, "official_fold": fold, "split": split,
                   **{column: name in labels for name, column in zip(data.CLASS_NAMES, data.LABEL_COLUMNS)},
                   "eligible_for_classification": True, "exclusion_reason": "", "labels_json": json.dumps(labels),
                   "image_path": relative, "image_sha256": digest(payload), "pixel_sha256": digest(image.tobytes()),
                   "width": 2240, "height": 3024, "image_bytes": len(payload), "clipped_point_count": 0}
            rows.append(row)
        groups = {"all_images.csv": rows, "classification_manifest.csv": rows, "classification_exclusions.csv": []}
        groups.update({f"splits/{split}.csv": [row for row in rows if row["split"] == split] for split in self.counts})
        for name, selected in groups.items():
            path = source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(selected)
        configuration = {"class_names": list(data.CLASS_NAMES), "pixels_per_mm": 8,
                         "lead_order": data.PREPROCESSING["lead_order"], "header_mm": 10, "row_pitch_mm": 30,
                         "footer_mm": 8, "left_margin_mm": 25, "right_margin_mm": 5,
                         "samples_per_lead": 5000, "sampling_rate_hz": 500, "speed_mm_per_second": 25}
        write_json(source / "protocol.json", {"binding": {"synthetic": True}, "configuration": configuration})
        summary = {"binding": {"synthetic": True}, "images": 4, "classification_eligible": 4,
                   "eligible_partition_counts": self.counts,
                   "manifest_sha256": {name: digest((source / name).read_bytes()) for name in groups}}
        write_json(source / "generation_summary.json", summary)
        write_json(source / "generation_state.json", {"status": "complete", "images": 4})
        write_json(source / "verification.json", {
            "status": "passed", "failures": [], "pages_decoded_and_hashed": 4,
            "all_output_files_accounted_for": True, "all_source_records_accounted_for": True,
            "manifest_hashes_verified": True, "official_patient_grouping_verified": True,
            "classification_target_mapping_verified": True, "duplicates_crossing_partitions": []})
        return source

    def fixture_cache(self):
        source = self.fixture_source()
        cache = self.root / "cache"
        metadata = data.prepare_cache(cache, workers=1, dataset_root=source)
        return source, cache, metadata

    def test_full_twelve_rows_boundary_samples_and_calibration_survive(self):
        page = np.full(data.PAGE_SHAPE, 255, np.uint8)
        page[:80] = 0
        page[2960:] = 0
        for lead in range(12):
            top = 80 + lead * 240
            page[top, 0] = 255 - (lead + 1)
            page[top + 239, 2239] = 255 - (lead + 13)
            page[top + 120, 96] = 0  # Calibration region is retained.
        original = page.copy()
        result = data.preprocess_page(page)
        self.assertEqual(result.shape, (12, 120, 1120))
        self.assertEqual(result.dtype, np.uint8)
        for lead in range(12):
            self.assertEqual(result[lead, 0, 0], lead + 1)
            self.assertEqual(result[lead, -1, -1], lead + 13)
            self.assertEqual(result[lead, 60, 48], 255)
            self.assertEqual(np.count_nonzero(result[lead]), 3)
        np.testing.assert_array_equal(page, original)

    def test_max_pool_preserves_darkest_member_with_correct_polarity(self):
        page = np.full(data.PAGE_SHAPE, 255, np.uint8)
        page[80:82, :2] = [[220, 180], [110, 40]]
        result = data.preprocess_page(page)
        self.assertEqual(result[0, 0, 0], 215)
        self.assertEqual(result[0, 0, 1], 0)
        self.assertTrue(result.flags.c_contiguous)

    def test_preprocessing_rejects_wrong_dimensions_color_and_dtype(self):
        for page in (np.zeros((3023, 2240), np.uint8), np.zeros((3024, 2240, 3), np.uint8),
                     np.zeros(data.PAGE_SHAPE, np.float32), None):
            with self.subTest(shape=getattr(page, "shape", None)), self.assertRaises(ValueError):
                data.preprocess_page(page)

    def test_boolean_parser_rejects_truthy_strings_and_preserves_false(self):
        self.assertIs(data._strict_bool("False"), False)
        self.assertIs(data._strict_bool("True"), True)
        for invalid in ("false", "0", "1", " TRUE", "", False, True, None):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                data._strict_bool(invalid)

    def test_source_loader_preserves_overlapping_targets_and_rejects_manifest_tampering(self):
        source = self.fixture_source()
        rows = data.load_source_rows(source)
        self.assertEqual([row["ecg_id"] for row in rows], [1, 2, 3, 4])
        self.assertEqual([rows[0][key] for key in data.LABEL_COLUMNS], [True, True, False, False, False])
        self.assertIs(rows[1]["label_NORM"], False)
        with (source / "splits/test.csv").open("a") as handle:
            handle.write("\n")
        with self.assertRaisesRegex(ValueError, "manifest hash mismatch"):
            data.load_source_rows(source)

    def test_source_rows_reject_patient_leakage_invalid_targets_and_paths(self):
        source = self.fixture_source()
        rows = data._read_csv(source / "classification_manifest.csv")
        changes = ((2, "patient_id", rows[0]["patient_id"]), (0, "label_STTC", "False "),
                   (0, "image_path", "../image.png"), (0, "eligible_for_classification", "False"),
                   (0, "official_fold", "9"), (0, "labels_json", '["NORM"]'))
        for index, key, value in changes:
            changed = [dict(row) for row in rows]
            changed[index][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                data._parse_rows(changed)

    def test_source_loader_requires_completed_verification(self):
        source = self.fixture_source()
        state = json.loads((source / "generation_state.json").read_text())
        state["status"] = "generated_pending_full_verification"
        write_json(source / "generation_state.json", state)
        with self.assertRaisesRegex(ValueError, "completed full verification"):
            data.load_source_rows(source)

    def test_train_only_statistics_ignore_even_extreme_test_statistics(self):
        count = np.prod(data.INPUT_SHAPE).item()
        training = [{"row": {"split": "train"}, "train_pixels": count,
                     "train_sum": mean * count, "train_sum_squares": mean * mean * count} for mean in (0.2, 0.6)]
        forbidden = {"row": {"split": "test"}, "train_pixels": 10**18,
                     "train_sum": float("nan"), "train_sum_squares": float("nan")}
        fitted = data._training_normalization(training + [forbidden])
        self.assertAlmostEqual(fitted["mean"], 0.4)
        self.assertAlmostEqual(fitted["std"], 0.2)
        self.assertEqual(fitted["training_pixels"], 2 * count)

    def test_end_to_end_lazy_dataset_normalization_targets_and_complete_reuse(self):
        source, cache, metadata = self.fixture_cache()
        self.assertEqual(metadata["counts"], self.counts)
        self.assertAlmostEqual(metadata["normalization"]["mean"], 64 / 255)
        self.assertAlmostEqual(metadata["normalization"]["std"], 64 / 255)
        train = data.CachedECGDataset(cache, "train")
        self.assertEqual(len(train), 2)
        first, target = train[0]
        self.assertEqual(tuple(first.shape), data.INPUT_SHAPE)
        self.assertEqual(first.dtype, torch.float32)
        self.assertEqual(target.dtype, torch.float32)
        np.testing.assert_array_equal(target.numpy(), [1, 1, 0, 0, 0])
        self.assertTrue(torch.all(first == -1).item())
        self.assertTrue(torch.all(train[1][0] == 1).item())
        raw = data.CachedECGDataset(cache, "validation", normalize=False)
        self.assertTrue(torch.all(raw[0][0] == 1).item())
        paths_before = {str(p): p.stat().st_mtime_ns for p in cache.rglob("*") if p.is_file()}
        self.assertEqual(data.prepare_cache(cache, workers=1, dataset_root=source), metadata)
        self.assertEqual(paths_before, {str(p): p.stat().st_mtime_ns for p in cache.rglob("*") if p.is_file()})

    def test_source_checksum_failure_keeps_failed_output_and_refuses_partial_resume(self):
        source = self.fixture_source()
        path = source / "images/train/ecg_00001.png"
        path.write_bytes(path.read_bytes() + b"tampered")
        cache = self.root / "cache"
        with self.assertRaisesRegex(ValueError, "Source image checksum"):
            data.prepare_cache(cache, workers=1, dataset_root=source)
        self.assertTrue((cache / "failure.json").exists())
        self.assertFalse((cache / "metadata.json").exists())
        with self.assertRaisesRegex(FileExistsError, "Incomplete/nonempty"):
            data.prepare_cache(cache, workers=1, dataset_root=source)

    def test_dataset_rejects_encoded_pixel_hash_and_wrong_shape(self):
        _, cache, _ = self.fixture_cache()
        dataset = data.CachedECGDataset(cache, "train")
        row = dataset.rows[0]
        path = cache / row["cache_path"]
        original = path.read_bytes()
        path.write_bytes(original + b"tampered")
        with self.assertRaisesRegex(ValueError, "encoded checksum"):
            dataset[0]
        path.write_bytes(original)
        row["cache_pixel_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "pixel checksum"):
            dataset[0]
        ok, encoded = cv2.imencode(".png", np.zeros((16, 16), np.uint8))
        self.assertTrue(ok)
        payload = encoded.tobytes()
        path.write_bytes(payload)
        row["cache_sha256"] = digest(payload)
        with self.assertRaisesRegex(ValueError, "Wrong cache image shape"):
            dataset[0]

    def test_cache_metadata_manifest_and_code_binding_are_checked(self):
        _, cache, _ = self.fixture_cache()
        manifest = cache / "cache_manifest.csv"
        original = manifest.read_bytes()
        manifest.write_bytes(original + b"\n")
        with self.assertRaisesRegex(ValueError, "Cache manifest hash"):
            data.CachedECGDataset(cache, "train")
        manifest.write_bytes(original)
        protocol = json.loads((cache / "protocol.json").read_text())
        metadata = json.loads((cache / "metadata.json").read_text())
        protocol["binding"]["code_sha256"] = "0" * 64
        metadata["binding"] = protocol["binding"]
        write_json(cache / "protocol.json", protocol)
        write_json(cache / "metadata.json", metadata)
        with self.assertRaisesRegex(ValueError, "binding changed"):
            data.CachedECGDataset(cache, "train")


if __name__ == "__main__":
    unittest.main()
