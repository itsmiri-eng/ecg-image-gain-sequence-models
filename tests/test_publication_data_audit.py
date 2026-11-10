"""Safety/integrity tests for the new non-mutating audit entry point."""
import hashlib
import json

import cv2
import numpy as np
import pytest

from src import audit_publication_data as audit


def test_existing_output_rejected_before_any_checks(tmp_path, monkeypatch):
    output = tmp_path / "verification.json"
    output.write_text("original frozen artifact")
    monkeypatch.setattr(audit, "_collect_report", lambda *args: pytest.fail("Existing output must fail before checking"))
    with pytest.raises(FileExistsError):
        audit.run_audit(output, "full")
    assert output.read_text() == "original frozen artifact"


def test_concurrent_output_creation_cannot_be_overwritten(tmp_path, monkeypatch):
    output = tmp_path / "audit.json"
    def collect(*args):
        output.write_text("other writer")
        return {"status": "passed", "mode": "metadata"}
    monkeypatch.setattr(audit, "_collect_report", collect)
    with pytest.raises(FileExistsError):
        audit.run_audit(output)
    assert output.read_text() == "other writer"


def test_failed_checks_write_no_success_report(tmp_path, monkeypatch):
    output = tmp_path / "audit.json"
    def fail(*args):
        raise ValueError("mismatched source")
    monkeypatch.setattr(audit, "_collect_report", fail)
    with pytest.raises(ValueError, match="mismatched source"):
        audit.run_audit(output)
    assert not output.exists()


def test_metadata_mode_remains_explicit_in_saved_report(tmp_path, monkeypatch):
    output = tmp_path / "audit.json"
    def collect(mode, workers):
        assert mode == "metadata"
        return {"status": "passed", "mode": mode, "ptbxl": {"full_file_checks_performed": False, "full_file_checks": None}}
    monkeypatch.setattr(audit, "_collect_report", collect)
    audit.run_audit(output)
    saved = json.loads(output.read_text())
    assert saved["mode"] == "metadata"
    assert saved["ptbxl"]["full_file_checks_performed"] is False
    assert saved["ptbxl"]["full_file_checks"] is None


@pytest.fixture
def cache_record(tmp_path):
    pixels = np.arange(1440 * 1120, dtype=np.uint8).reshape(1440, 1120)
    ok, encoded = cv2.imencode(".png", pixels)
    assert ok
    payload = encoded.tobytes()
    (tmp_path / "image.png").write_bytes(payload)
    return tmp_path, {"ecg_id": 1, "split": "train", "cache_path": "image.png",
                      "cache_sha256": hashlib.sha256(payload).hexdigest(),
                      "cache_pixel_sha256": hashlib.sha256(pixels.tobytes()).hexdigest()}


def test_cache_scan_checks_encoded_and_decoded_hashes(cache_record):
    root, row = cache_record
    before = (root / row["cache_path"]).read_bytes()
    result = audit._cache_record(row, root)
    assert result["pixels"] == 1440 * 1120
    assert (root / row["cache_path"]).read_bytes() == before
    with pytest.raises(ValueError, match="encoded hash"):
        audit._cache_record({**row, "cache_sha256": "0" * 64}, root)
    with pytest.raises(ValueError, match="decoded-pixel hash"):
        audit._cache_record({**row, "cache_pixel_sha256": "0" * 64}, root)


def test_normalization_excludes_validation_and_test(cache_record):
    root, row = cache_record
    training = audit._cache_record(row, root)
    validation = audit._cache_record({**row, "split": "validation"}, root)
    test = audit._cache_record({**row, "split": "test"}, root)
    assert "total" not in validation and "total" not in test
    assert audit._training_moments([training]) == audit._training_moments([training, validation, test])


def test_broken_symlink_output_is_also_rejected(tmp_path, monkeypatch):
    output = tmp_path / "audit.json"
    try:
        output.symlink_to(tmp_path / "missing.json")
    except OSError:  # Windows without Developer Mode cannot create symlinks
        pytest.skip("symlinks are not permitted for this user")
    monkeypatch.setattr(audit, "_collect_report", lambda *args: pytest.fail("Symlink output must fail before checking"))
    with pytest.raises(FileExistsError):
        audit.run_audit(output)
    assert output.is_symlink()
