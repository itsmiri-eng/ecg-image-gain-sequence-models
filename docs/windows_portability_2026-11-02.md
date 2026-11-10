# Windows portability fixes (2 Nov 2026, NVIDIA PC)

The code was written and tested on macOS. On the Windows PC, 22 of 219 tests failed and the three
v1 seed-42 runs (CNN–LSTM, temporal conv, hybrid ViT) could no longer be re-verified. None of this
changed any result: no model, prediction, metric or dataset file was touched. What was fixed:

| Problem on Windows | Fix | Files |
|---|---|---|
| Lock key compared with a `\` path; every PC run stopped before its log row | `.as_posix()` (see [record recovery](../experiments/record_recovery/README.md)) | `src/baseline_preservation.py` |
| Verifiers built relative paths with `\`, saved records use `/` | `X.relative_to(Y).as_posix()` everywhere (identical on macOS) | `src/verify_ptbxl_run.py`, `src/verify_ptbxl_spatiotemporal.py`, `src/verify_ptbxl_hybrid_vit.py`, `src/compare_ptbxl_models.py`, two verification tests, `tests/test_ptbxl_comparison.py` |
| The v1 cache and training metadata recorded absolute Mac folders | Verifiers accept a recorded path only if it names the same file below the project folder (`relocate`, `located` in `src/portable_cache.py`; unchanged behaviour on the original computer) | same verifiers |
| Publication summary hashed `\n` text but wrote `\r\n` | `newline=""` when writing | `src/publication_summary.py` |
| No `git` on the PC | Test patches the runner's commit lookup to `None` (what it records without a repository) | `tests/test_ptbxl_hybrid_vit_runner.py` |
| Symlinks need Windows Developer Mode | Two symlink tests skip when symlinks cannot be made | `tests/test_publication_summary.py`, `tests/test_publication_data_audit.py` |
| Strict v1 loader needs the Mac folder | Test still checks relocation; skips only the strict comparison off the original computer | `tests/test_portable_cache.py` |
| Batch-order rounding of 6e-8 in one CPU test | Compare with the runners' own 1e-6 replay limit | `tests/test_ptbxl_cnn_lstm.py` |

## Approved source revisions

Runs record SHA-256 hashes of their source files, and later runs also recorded earlier verifiers as
protected evidence. Four of those files were edited above. `src/approved_source_revisions.py`
lists each (file, recorded hash) → approved new hash; verifiers accept exactly that pair and
nothing else, and a test fails if an approved file changes again without updating the list. Each
run's own `source_snapshot/` keeps the original bytes and is still checked against the recorded
hash.

Not approved because nothing verifies them: `src/run_ptbxl_experiment.py` and
`src/training/ptbxl_trainer.py` differ from the versions the Mac-trained `ptbxl_v2_cnn_pool_seed42`
recorded. That is the `--device cuda` change made on the Mac before the move, documented in
[MOVE_TO_NVIDIA_PC.md](MOVE_TO_NVIDIA_PC.md); every PC run recorded the new hashes.

## Verified after the fixes

- Tests: 218 passed, 3 skipped (the two symlink tests and the strict Mac-only cache comparison).
- The three v1 seed-42 runs pass full re-verification on the PC (run in memory; their saved
  `verification.json` files were not rewritten). Every artifact hash from the original certificates
  matches; the only extra files are report figures and reviews added after certification.

## Known, harmless difference

`src/common.write_json` writes text with the OS line ending, so JSON files written on the PC end
lines with `\r\n` and Mac-written ones with `\n`. Parsed content is identical and every hash is taken
of the bytes actually written, so checks stay consistent. `src/common.py` is hash-bound in every
run and was deliberately not changed.
