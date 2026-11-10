"""Small synthetic checks of PTB labels, patient isolation and source binding."""
import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.ptbxl_image_manifest import CLASS_NAMES, prepare_manifest


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PTBXLManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dataset = self.root / "dataset"
        self.dataset.mkdir()
        self.audit = self.root / "prior_audit.json"

    def fixture(self, specs=None):
        specs = specs or [
            (3, "30.0", 10, "{'SR': 0.0}"),
            (1, "10.0", 1, "{'NORM': 100.0, 'ASMI': 0.0, 'SR': 0.0}"),
            (2, "20.0", 9, "{'LVH': 50.0, 'NDT': 0.0, 'LAFB': 0.0}"),
        ]
        statements = [
            ("NORM", "1.0", "NORM"), ("ASMI", "1.0", "MI"),
            ("NDT", "1.0", "STTC"), ("LAFB", "1.0", "CD"),
            ("LVH", "1.0", "HYP"), ("SR", "", ""),
        ]
        with (self.dataset / "scp_statements.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["", "diagnostic", "diagnostic_class"])
            writer.writerows(statements)
        with (self.dataset / "ptbxl_database.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["ecg_id", "patient_id", "strat_fold", "scp_codes", "filename_hr", "filename_lr"])
            writer.writeheader()
            for ecg, patient, fold, codes in specs:
                row = {"ecg_id": ecg, "patient_id": patient, "strat_fold": fold, "scp_codes": codes}
                for key, rate, suffix in (("filename_hr", 500, "hr"), ("filename_lr", 100, "lr")):
                    relative = f"records{rate}/00000/{ecg:05d}_{suffix}"
                    row[key] = relative
                    for extension in (".dat", ".hea"):
                        path = self.dataset / (relative + extension)
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(b"synthetic waveform fixture")
                writer.writerow(row)
        paths = sorted(p for p in self.dataset.rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt")
        (self.dataset / "SHA256SUMS.txt").write_text("".join(f"{sha(p)} {p.relative_to(self.dataset).as_posix()}\n" for p in paths))
        self.rebind_audit(len(specs))

    def rebind_audit(self, n):
        hashes = {name: sha(self.dataset / name) for name in ("ptbxl_database.csv", "scp_statements.csv", "SHA256SUMS.txt")}
        self.audit.write_text(json.dumps({"source_hashes": hashes, "record_count": n}))

    def test_zero_confidence_multilabel_and_unmapped_records_preserved(self):
        self.fixture()
        result = prepare_manifest(self.dataset, self.audit)
        self.assertEqual(CLASS_NAMES, ("NORM", "MI", "STTC", "CD", "HYP"))
        self.assertEqual([row["ecg_id"] for row in result.rows], [1, 2, 3])
        first, second, third = result.rows
        self.assertEqual(first["patient_id"], "10")
        self.assertTrue(first["label_MI"])
        self.assertTrue(first["label_NORM"])
        self.assertEqual(json.loads(first["scp_codes_json"])["ASMI"], 0.0)
        self.assertEqual(json.loads(second["labels_json"]), ["STTC", "CD", "HYP"])
        self.assertFalse(third["eligible_for_classification"])
        self.assertEqual(third["exclusion_reason"], "no_mapped_diagnostic_superclass")
        self.assertEqual(third["split"], "test")
        self.assertEqual(result.summary["records_to_render"], 3)
        self.assertEqual(result.summary["classification_eligible_records"], 2)
        self.assertEqual(result.summary["partitions"]["test"]["classification_eligible_records"], 0)

    def test_same_patient_multiple_records_within_fold_allowed(self):
        self.fixture([(1, "10", 1, "{'NORM': 100}"), (2, "10.0", 1, "{'ASMI': 100}")])
        result = prepare_manifest(self.dataset, self.audit)
        self.assertEqual(result.summary["all_source_patients"], 1)
        self.assertEqual(result.summary["records_to_render"], 2)

    def test_patient_overlap_rejected_even_between_two_training_folds(self):
        self.fixture([(1, "10", 1, "{'NORM': 100}"), (2, "10.0", 2, "{'ASMI': 100}")])
        with self.assertRaisesRegex(ValueError, "multiple official folds"):
            prepare_manifest(self.dataset, self.audit)

    def test_duplicate_ecg_id_rejected(self):
        self.fixture([(1, "10", 1, "{'NORM': 100}"), (1, "20", 9, "{'ASMI': 100}")])
        with self.assertRaisesRegex(ValueError, "Duplicate ECG ID"):
            prepare_manifest(self.dataset, self.audit)

    def test_invalid_fold_and_unknown_code_rejected(self):
        for fold, codes, error in ((11, "{'NORM': 100}", "outside 1..10"), (1, "{'UNKNOWN': 100}", "Unmapped SCP")):
            with self.subTest(fold=fold, codes=codes):
                self.fixture([(1, "10", fold, codes)])
                with self.assertRaisesRegex(ValueError, error):
                    prepare_manifest(self.dataset, self.audit)

    def test_metadata_tamper_rejected_against_prior_audit(self):
        self.fixture()
        with (self.dataset / "ptbxl_database.csv").open("a") as handle:
            handle.write("\n")
        with self.assertRaisesRegex(ValueError, "prior verified audit"):
            prepare_manifest(self.dataset, self.audit)

    def test_rebinding_audit_cannot_bypass_shipped_metadata_hash(self):
        self.fixture()
        with (self.dataset / "ptbxl_database.csv").open("a") as handle:
            handle.write("\n")
        self.rebind_audit(3)
        with self.assertRaisesRegex(ValueError, "shipped checksum"):
            prepare_manifest(self.dataset, self.audit)

    def test_missing_waveform_rejected_without_loading_signals(self):
        self.fixture()
        (self.dataset / "records500/00000/00001_hr.dat").unlink()
        with self.assertRaisesRegex(ValueError, "missing from disk"):
            prepare_manifest(self.dataset, self.audit)

    def test_deterministic_read_only_result_and_source_code_retention(self):
        self.fixture()
        before = {p: sha(p) for p in self.dataset.rglob("*") if p.is_file()}
        first = prepare_manifest(self.dataset, self.audit)
        second = prepare_manifest(self.dataset, self.audit)
        self.assertEqual(first, second)
        self.assertEqual(before, {p: sha(p) for p in self.dataset.rglob("*") if p.is_file()})
        self.assertIn("SR", json.loads(first.rows[0]["scp_codes_json"]))
        self.assertEqual(first.rows[0]["filename_hr"], "records500/00000/00001_hr")
        self.assertEqual(first.summary["records_filtered_from_rendering"], 0)


if __name__ == "__main__":
    unittest.main()
