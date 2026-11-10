"""Temporary-only generator checks: provenance, safe resume and row preservation."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from src.generate_ptbxl_images import _manifest_row, _render_job, _verify_job


class GenerationWorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.output = self.root / "derived"
        self.row = {
            "ecg_id": 1, "patient_id": "10", "official_fold": 1, "split": "train",
            "filename_hr": "records500/00000/00001_hr", "filename_lr": "records100/00000/00001_lr",
            "label_NORM": True, "label_MI": True, "label_STTC": False,
            "label_CD": False, "label_HYP": False,
            "labels_json": '["NORM","MI"]', "scp_codes_json": '{"ASMI":0.0,"NORM":100.0}',
            "eligible_for_classification": True, "exclusion_reason": "",
        }
        self.config = {"renderer": "tiny_mock_renderer"}
        self.config_hash = hashlib.sha256(json.dumps(self.config).encode()).hexdigest()
        self.checksums = {}
        for extension in (".hea", ".dat"):
            name = self.row["filename_hr"] + extension
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            data = ("synthetic source " + extension).encode()
            path.write_bytes(data)
            self.checksums[name] = hashlib.sha256(data).hexdigest()
        self.image = np.arange(48, dtype=np.uint8).reshape(6, 8)
        success, encoded = cv2.imencode(".png", self.image)
        self.assertTrue(success)
        self.payload = encoded.tobytes()
        self.image_path = self.output / "images/train/ecg_00001.png"
        self.sidecar_path = self.output / "provenance/train/ecg_00001.json"
        self.geometry = {
            "width": 8, "height": 6, "gain_mm_per_mv": 10.0,
            "seconds_per_pixel": .005, "mv_per_pixel": .025,
            "quality_flags": [], "flat_leads": [], "clipped_point_count": 0,
            "nonfinite_point_count": 0, "out_of_row_point_count": 0,
            "source_points_plotted": 60000,
            "lead_panels": [{"lead": lead, "sample_start": 0, "sample_stop_exclusive": 5000}
                            for lead in ("I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6")],
        }

    def job(self, row=None):
        return (row or self.row, str(self.source), str(self.output), self.config, self.checksums, self.config_hash)

    def render(self, row=None):
        with patch("src.generate_ptbxl_images.load_waveform", return_value=(np.zeros((5000, 12)), {"sampling_rate_hz": 500})) as load:
            with patch("src.generate_ptbxl_images.render_page", return_value=(self.image, self.geometry)):
                with patch("src.generate_ptbxl_images.encode_png", return_value=self.payload):
                    result = _render_job(self.job(row))
        return result, load

    def save_metadata(self, metadata):
        self.sidecar_path.write_text(json.dumps(metadata))

    def verify_job(self, row=None, sidecar_hash=None):
        return (row or self.row, str(self.output), str(self.source), self.config_hash,
                self.checksums, sidecar_hash or hashlib.sha256(self.sidecar_path.read_bytes()).hexdigest())

    def test_new_page_preserves_all_multilabel_source_fields(self):
        result, load = self.render()
        self.assertTrue(result["ok"], result)
        self.assertFalse(result["reused"])
        load.assert_called_once()
        meta = result["metadata"]
        self.assertEqual(meta["source_record"], self.row)
        self.assertEqual(meta["source_sha256"], self.checksums)
        self.assertEqual(meta["configuration_sha256"], self.config_hash)
        self.assertEqual(meta["image_sha256"], hashlib.sha256(self.payload).hexdigest())
        manifest = _manifest_row(meta)
        self.assertEqual({key: manifest[key] for key in self.row}, self.row)
        self.assertTrue(manifest["label_NORM"] and manifest["label_MI"])
        self.assertEqual(manifest["source_waveform_sha256"], self.checksums[self.row["filename_hr"] + ".dat"])

    def test_classification_ineligible_record_is_still_rendered_and_logged(self):
        row = copy.deepcopy(self.row)
        row.update({key: False for key in row if key.startswith("label_")})
        row.update(labels_json="[]", scp_codes_json='{"SR":0.0}', eligible_for_classification=False,
                   exclusion_reason="no_mapped_diagnostic_superclass")
        result, _ = self.render(row)
        self.assertTrue(result["ok"], result)
        self.assertTrue(self.image_path.is_file())
        manifest = _manifest_row(result["metadata"])
        self.assertFalse(manifest["eligible_for_classification"])
        self.assertEqual(manifest["exclusion_reason"], "no_mapped_diagnostic_superclass")

    def test_source_checksum_failure_prevents_decode_and_output(self):
        (self.source / (self.row["filename_hr"] + ".dat")).write_bytes(b"changed")
        result, load = self.render()
        self.assertFalse(result["ok"])
        self.assertIn("checksum changed", result["error"])
        load.assert_not_called()
        self.assertFalse(self.image_path.exists())
        self.assertFalse(self.sidecar_path.exists())

    def test_valid_resume_rehashes_sources_and_reuses_existing_page(self):
        first, _ = self.render()
        self.assertTrue(first["ok"])
        bytes_before = self.image_path.read_bytes(), self.sidecar_path.read_bytes()
        second, load = self.render()
        self.assertTrue(second["ok"], second)
        self.assertTrue(second["reused"])
        load.assert_not_called()
        self.assertEqual(bytes_before, (self.image_path.read_bytes(), self.sidecar_path.read_bytes()))
        (self.source / (self.row["filename_hr"] + ".hea")).write_bytes(b"changed after first rendering")
        third, load = self.render()
        self.assertFalse(third["ok"])
        load.assert_not_called()
        self.assertEqual(bytes_before, (self.image_path.read_bytes(), self.sidecar_path.read_bytes()))

    def test_resume_refuses_changed_source_label_or_configuration(self):
        result, _ = self.render()
        self.assertTrue(result["ok"])
        changed_row = {**self.row, "label_MI": False}
        changed, load = self.render(changed_row)
        self.assertFalse(changed["ok"])
        load.assert_not_called()
        metadata = result["metadata"]
        metadata["configuration_sha256"] = "0" * 64
        self.save_metadata(metadata)
        changed, load = self.render()
        self.assertFalse(changed["ok"])
        load.assert_not_called()

    def test_resume_refuses_modified_encoded_image(self):
        result, _ = self.render()
        self.assertTrue(result["ok"])
        self.image_path.write_bytes(b"altered page")
        result, load = self.render()
        self.assertFalse(result["ok"])
        load.assert_not_called()
        self.assertEqual(self.image_path.read_bytes(), b"altered page")

    def test_resume_refuses_wrong_sidecar_path_without_rewriting_page(self):
        rendered, _ = self.render()
        self.assertTrue(rendered["ok"])
        metadata = copy.deepcopy(rendered["metadata"])
        metadata["sidecar_path"] = "provenance/train/ecg_99999.json"
        self.save_metadata(metadata)
        before = (self.image_path.read_bytes(), self.sidecar_path.read_bytes())
        result, load = self.render()
        self.assertFalse(result["ok"])
        self.assertIn("does not match the frozen generation protocol", result["error"])
        load.assert_not_called()
        self.assertEqual(before, (self.image_path.read_bytes(), self.sidecar_path.read_bytes()))

    def test_existing_orphan_image_only_adopted_if_deterministic_bytes_match(self):
        self.image_path.parent.mkdir(parents=True)
        self.image_path.write_bytes(b"unverified different image")
        failure, load = self.render()
        self.assertFalse(failure["ok"])
        self.assertIn("differs from deterministic", failure["error"])
        self.assertFalse(self.sidecar_path.exists())
        self.assertEqual(self.image_path.read_bytes(), b"unverified different image")
        self.image_path.write_bytes(self.payload)
        success, load = self.render()
        self.assertTrue(success["ok"], success)
        self.assertFalse(success["reused"])
        self.assertTrue(self.sidecar_path.is_file())
        load.assert_called_once()

    def test_verification_binds_full_source_row_and_returns_exact_manifest(self):
        rendered, _ = self.render()
        self.assertTrue(rendered["ok"])
        result = _verify_job(self.verify_job())
        self.assertTrue(result["ok"], result)
        restored = result["manifest_row"]
        self.assertEqual({key: restored[key] for key in self.row}, self.row)
        self.assertEqual(restored["provenance_sha256"], hashlib.sha256(self.sidecar_path.read_bytes()).hexdigest())
        wrong_label = {**self.row, "label_MI": False}
        result = _verify_job(self.verify_job(row=wrong_label))
        self.assertFalse(result["ok"])
        self.assertIn("source identity mismatch", result["error"])

    def test_verification_rejects_changed_sidecar_bytes_before_trusting_fields(self):
        rendered, _ = self.render()
        self.assertTrue(rendered["ok"])
        original_hash = hashlib.sha256(self.sidecar_path.read_bytes()).hexdigest()
        self.sidecar_path.write_bytes(self.sidecar_path.read_bytes() + b"\n")
        result = _verify_job(self.verify_job(sidecar_hash=original_hash))
        self.assertFalse(result["ok"])
        self.assertIn("provenance hash mismatch", result["error"])

    def test_verification_rejects_rebound_wrong_configuration_or_source_hash(self):
        rendered, _ = self.render()
        self.assertTrue(rendered["ok"])
        for field, value in (("configuration_sha256", "0" * 64), ("source_sha256", {})):
            with self.subTest(field=field):
                metadata = copy.deepcopy(rendered["metadata"])
                metadata[field] = value
                self.save_metadata(metadata)
                # Rehash to exercise semantic binding in addition to sidecar-byte binding.
                result = _verify_job(self.verify_job())
                self.assertFalse(result["ok"])
                self.assertIn("source/configuration binding mismatch", result["error"])

    def test_verification_rejects_alternate_paths_even_with_matching_saved_hash(self):
        rendered, _ = self.render()
        self.assertTrue(rendered["ok"])
        for field in ("image_path", "sidecar_path"):
            with self.subTest(field=field):
                metadata = copy.deepcopy(rendered["metadata"])
                metadata[field] = "../outside-output.png"
                self.save_metadata(metadata)
                result = _verify_job(self.verify_job())
                self.assertFalse(result["ok"])
                self.assertIn("path mismatch", result["error"])

    def test_verification_rejects_duplicate_or_reordered_lead_panels(self):
        rendered, _ = self.render()
        self.assertTrue(rendered["ok"])
        for mode in ("duplicate_lead", "reordered_leads"):
            with self.subTest(mode=mode):
                metadata = copy.deepcopy(rendered["metadata"])
                panels = metadata["lead_panels"]
                if mode == "duplicate_lead":
                    panels[1]["lead"] = panels[0]["lead"]
                else:
                    panels[0], panels[1] = panels[1], panels[0]
                # Twelve complete-length panels remain: only identity/order is wrong.
                self.assertEqual(len(panels), 12)
                self.assertTrue(all(panel["sample_stop_exclusive"] == 5000 for panel in panels))
                self.save_metadata(metadata)
                result = _verify_job(self.verify_job())
                self.assertFalse(result["ok"])
                self.assertIn("Lead names/order differ", result["error"])

    def test_verification_rehashes_original_source_after_generation(self):
        rendered, _ = self.render()
        self.assertTrue(rendered["ok"])
        (self.source / (self.row["filename_hr"] + ".dat")).write_bytes(b"changed after rendering")
        result = _verify_job(self.verify_job())
        self.assertFalse(result["ok"])
        self.assertIn("changed after generation", result["error"])

    def test_verification_rejects_modified_image_and_pixel_hash(self):
        rendered, _ = self.render()
        self.assertTrue(rendered["ok"])
        self.image_path.write_bytes(self.payload + b"trailing change")
        result = _verify_job(self.verify_job())
        self.assertFalse(result["ok"])
        self.assertIn("Encoded image hash mismatch", result["error"])
        self.image_path.write_bytes(self.payload)
        metadata = copy.deepcopy(rendered["metadata"])
        metadata["pixel_sha256"] = "0" * 64
        self.save_metadata(metadata)
        result = _verify_job(self.verify_job())
        self.assertFalse(result["ok"])
        self.assertIn("Decoded pixel hash mismatch", result["error"])


if __name__ == "__main__":
    unittest.main()
