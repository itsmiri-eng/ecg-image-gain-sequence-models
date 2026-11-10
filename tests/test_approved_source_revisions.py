"""Verifiers accept only the exact documented source revision, nothing else."""
from src.approved_source_revisions import APPROVED, current_source_sha256
from src.common import ROOT, sha256_file

OLD = "76d5b4ff9f383c3a6a5a3ab8bdae81ca83c36ae8ab11cc64d08cc3cbb474f673"


def test_only_the_exact_recorded_hash_maps_to_the_approved_revision():
    new = APPROVED[("src/baseline_preservation.py", OLD)]
    assert current_source_sha256("src/baseline_preservation.py", OLD) == new
    assert current_source_sha256("src/baseline_preservation.py", "0" * 64) == "0" * 64
    assert current_source_sha256("src/common.py", OLD) == OLD


def test_every_approved_revision_is_the_file_in_this_project():
    # Fails if an approved file is edited again without updating (and documenting) its approval.
    for (name, _), approved in APPROVED.items():
        assert sha256_file(ROOT / name) == approved, name
