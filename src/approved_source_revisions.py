"""Documented source edits made after runs recorded their source hashes, which verifiers accept.

A verifier checks that each source file a run recorded still has the recorded SHA-256. A file listed
here may instead have exactly the approved later hash; any other content still fails. The run's own
source_snapshot copy keeps the original bytes and is checked against the recorded hash as before.
"""

APPROVED = {
    # 2 Nov 2026, Windows PC: lock-key path written with "/" instead of the OS separator (one line).
    # Reason and evidence: experiments/record_recovery/README.md
    ("src/baseline_preservation.py", "76d5b4ff9f383c3a6a5a3ab8bdae81ca83c36ae8ab11cc64d08cc3cbb474f673"):
        "73b3779df26355ec31ca51d96bfb1f5666ce4aadc4e65cbd60b0515c644f041a",
    # 2 Nov 2026, Windows PC: verifiers and comparison made portable. Relative paths are written with "/",
    # absolute paths recorded on the Mac are mapped into this project (src/portable_cache.py), and the
    # approved revisions above are accepted. Later runs recorded these files as protected prior evidence.
    # Reason and evidence: docs/windows_portability_2026-11-02.md
    ("src/compare_ptbxl_models.py", "cf7099c1f960f5325ce0565a61209e20930fcfbd4ae66d12a76842c584fac8c9"):
        "817a4577f29a1c41e8a1ff78714e77e04d9d4838e0837d19492d8d91e7596741",
    ("src/verify_ptbxl_run.py", "524d2c7d375fe046974ab7ed28db40ccb78ae22eedf2b936c1299eb3f023dde2"):
        "d0cd9b01ce515eb0b996ed1ced8572ee591f485b792a879df46a3433521a9b82",
    ("src/verify_ptbxl_spatiotemporal.py", "592b896cca4c8851006ce1d9b4aacb3e28a738b34b9596ebf1d57bbf1e7ea583"):
        "74608762b134bdaf5e74ef97f8933ede8c3c802b01e7d71cf283a597c7d53add",
}


def current_source_sha256(name, recorded):
    """The hash the file `name` must have today, given the hash a run recorded for it."""
    return APPROVED.get((name, recorded), recorded)
