#!/usr/bin/env python
# coding: utf-8
"""
PTB-XL Image Generation (no torch required).
Run in the ecgimg conda environment.
Generates one paper ECG image per selected PTB-XL record and writes
ptbxl_manifest.csv beside this script.

Label mapping uses the official scp_statements.csv diagnostic_class field.
MI vs History_MI is separated using infarction_stadium1/stadium2 only.
Each of the 600 selected records belongs to a unique patient.
"""

# ============================================================
# CONFIGURATION — edit these paths before running
# ============================================================
from pathlib import Path

# Folder containing ptbxl_database.csv, scp_statements.csv, and records500/
# NOTE: the dataset is currently at the path below, which differs from the
# originally intended ~/Downloads/ptb-xl-... location. Update if you move it.
PTBXL_ROOT = Path.home() / "Downloads" / \
    "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"

# Where generated PNG images are stored. Must be empty (or absent) before running.
OUTPUT_DIR = Path.home() / "Downloads" / "ptbxl_generated_images"

# Full absolute path to gen_ecg_images_from_data_batch.py
GEN_ECG_SCRIPT = Path.home() / "ecg-image-kit" / "codes" / \
    "ecg-image-generator" / "gen_ecg_images_from_data_batch.py"

SAMPLES_PER_CLASS = 150
SEED = 42
# ============================================================

# Manifest is always saved beside this script regardless of cwd
PROJECT_DIR = Path(__file__).resolve().parent
MANIFEST    = PROJECT_DIR / "ptbxl_manifest.csv"

import ast
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import random

random.seed(SEED)
np.random.seed(SEED)

CLASS_NAMES = ['Normal', 'MI', 'Abnormal', 'History_MI']

# ── Pre-flight checks ─────────────────────────────────────────
CSV_PATH      = PTBXL_ROOT / "ptbxl_database.csv"
SCP_STMTS_PATH = PTBXL_ROOT / "scp_statements.csv"

if not PTBXL_ROOT.exists():
    sys.exit(
        f"\nERROR: PTB-XL root directory not found:\n  {PTBXL_ROOT}\n"
        "Set PTBXL_ROOT in the CONFIGURATION section."
    )
if not CSV_PATH.exists():
    sys.exit(
        f"\nERROR: ptbxl_database.csv not found:\n  {CSV_PATH}"
    )
if not SCP_STMTS_PATH.exists():
    sys.exit(
        f"\nERROR: scp_statements.csv not found:\n  {SCP_STMTS_PATH}"
    )
if not GEN_ECG_SCRIPT.exists():
    sys.exit(
        f"\nERROR: Generator script not found:\n  {GEN_ECG_SCRIPT}\n"
        "Set GEN_ECG_SCRIPT in the CONFIGURATION section."
    )

# ── Refuse to mix with an existing nonempty output folder ─────
if OUTPUT_DIR.exists():
    existing_pngs = list(OUTPUT_DIR.rglob("*.png"))
    if existing_pngs:
        sys.exit(
            f"\nERROR: Output directory already contains {len(existing_pngs)} PNG file(s):\n"
            f"  {OUTPUT_DIR}\n"
            "Delete all existing images before running, or set OUTPUT_DIR to a new empty folder.\n"
            "Mixing images from different generation runs invalidates the manifest."
        )

# ── Load scp_statements.csv and derive official MI code set ───
scp_stmts = pd.read_csv(SCP_STMTS_PATH, index_col=0)
MI_CODES  = set(scp_stmts[scp_stmts["diagnostic_class"] == "MI"].index)
if not MI_CODES:
    sys.exit("\nERROR: No MI codes found in scp_statements.csv.")
print(f"Official MI codes ({len(MI_CODES)}): {sorted(MI_CODES)}")

# ── Stadium constants for MI vs History_MI separation ─────────
STADIUM_HISTORY = {"Stadium III"}
STADIUM_ACTIVE  = {"Stadium I", "Stadium I-II", "Stadium II"}
STADIUM_EXCLUDE = {"unknown", "Stadium II-III"}

# ── Load and classify PTB-XL records ──────────────────────────
print("\nLoading PTB-XL metadata...")
db = pd.read_csv(CSV_PATH)

def parse_scp_keys(s):
    """Return the set of all SCP code keys, regardless of value (incl. 0)."""
    try:
        return set(ast.literal_eval(s).keys())
    except Exception:
        return set()

db["scp_keys"] = db["scp_codes"].apply(parse_scp_keys)
db["has_mi"]   = db["scp_keys"].apply(lambda k: bool(k & MI_CODES))

def classify_stadium(row):
    """
    Classify an MI-coded record as MI, History_MI, or exclude based on
    infarction_stadium1 and infarction_stadium2.
    - History_MI: all usable stage values are exactly 'Stadium III'.
    - MI:         all usable stage values are in {Stadium I, I-II, II}.
    - exclude:    no stage, unknown, transitional, conflicting, or mixed stages.
    """
    stages = set()
    if pd.notna(row["infarction_stadium1"]):
        stages.add(row["infarction_stadium1"])
    if pd.notna(row["infarction_stadium2"]):
        stages.add(row["infarction_stadium2"])

    if not stages:
        return "exclude_no_stage"
    if stages & STADIUM_EXCLUDE:
        return "exclude_bad_stage"
    known = STADIUM_HISTORY | STADIUM_ACTIVE
    if stages - known:
        return "exclude_unknown_stage"
    has_history = bool(stages & STADIUM_HISTORY)
    has_active  = bool(stages & STADIUM_ACTIVE)
    if has_history and has_active:
        return "exclude_conflict"
    if has_history:
        return "History_MI"
    if has_active:
        return "MI"
    return "exclude_other"

mi_records = db[db["has_mi"]].copy()
mi_records["mi_class"] = mi_records.apply(classify_stadium, axis=1)

# ── Build class pools ─────────────────────────────────────────
curr_mi_pool  = mi_records[mi_records["mi_class"] == "MI"].copy()
hist_mi_pool  = mi_records[mi_records["mi_class"] == "History_MI"].copy()

normal_idx   = db["scp_keys"].apply(lambda k: "NORM" in k) & \
               ~db["has_mi"] & \
               db["scp_keys"].apply(lambda k: k <= {"NORM", "SR"})
normal_pool  = db[normal_idx].copy()

abnormal_pool = db[
    ~db["has_mi"] &
    ~db.index.isin(set(normal_pool.index)) &
    db["scp_keys"].apply(bool)
].copy()

# Assign integer class labels
curr_mi_pool ["label"] = 1
hist_mi_pool ["label"] = 3
normal_pool  ["label"] = 0
abnormal_pool["label"] = 2

# ── Print pool sizes and exclusion summary ────────────────────
print("\nFull eligible pool sizes:")
for name, pool in [("Normal", normal_pool), ("MI", curr_mi_pool),
                   ("Abnormal", abnormal_pool), ("History_MI", hist_mi_pool)]:
    print(f"  {name:<12}  records: {len(pool):>5}  "
          f"unique patients: {pool['patient_id'].nunique():>5}")

excl = mi_records[~mi_records["mi_class"].isin(["MI", "History_MI"])]
print(f"\nMI-coded records excluded by stadium rule: {len(excl)}")
for reason, grp in excl.groupby("mi_class"):
    print(f"  {reason}: {len(grp)} records")

# ── Eligibility gate ──────────────────────────────────────────
for name, pool in [("Normal", normal_pool), ("MI", curr_mi_pool),
                   ("Abnormal", abnormal_pool), ("History_MI", hist_mi_pool)]:
    n_pat = pool["patient_id"].nunique()
    if n_pat < SAMPLES_PER_CLASS:
        sys.exit(
            f"\nERROR: {name} has only {n_pat} unique patients "
            f"(need {SAMPLES_PER_CLASS}). Cannot proceed."
        )

# ── Select 150 per class, globally unique patients ────────────
# Priority for cross-class conflict resolution: MI > History_MI > Abnormal > Normal

def dedup_by_patient(pool, seed=SEED):
    """One record per patient, chosen reproducibly."""
    return (pool.sample(frac=1, random_state=seed)
                .drop_duplicates(subset="patient_id", keep="first"))

curr_mi_ded  = dedup_by_patient(curr_mi_pool)
hist_mi_ded  = dedup_by_patient(hist_mi_pool)
abnormal_ded = dedup_by_patient(abnormal_pool)
normal_ded   = dedup_by_patient(normal_pool)

used_patients: set = set()

def select_globally_unique(pool_ded, n, name, seed=SEED):
    global used_patients
    available = pool_ded[~pool_ded["patient_id"].isin(used_patients)]
    if len(available) < n:
        sys.exit(
            f"\nERROR: Only {len(available)} globally unique patients available "
            f"for {name} after cross-class deduplication (need {n})."
        )
    selected = available.sample(n=n, random_state=seed)
    used_patients.update(selected["patient_id"].tolist())
    return selected

sel_mi   = select_globally_unique(curr_mi_ded,  SAMPLES_PER_CLASS, "MI")
sel_hist = select_globally_unique(hist_mi_ded,  SAMPLES_PER_CLASS, "History_MI")
sel_ab   = select_globally_unique(abnormal_ded, SAMPLES_PER_CLASS, "Abnormal")
sel_norm = select_globally_unique(normal_ded,   SAMPLES_PER_CLASS, "Normal")

selected = pd.concat([sel_mi, sel_hist, sel_ab, sel_norm]).reset_index(drop=True)

# Sanity checks on selection
assert len(selected) == 4 * SAMPLES_PER_CLASS, "Wrong total record count"
assert selected["patient_id"].nunique() == len(selected), \
    "Duplicate patient_ids in selection"
assert selected["ecg_id"].nunique() == len(selected), \
    "Duplicate ecg_ids in selection"
assert selected["filename_hr"].nunique() == len(selected), \
    "Duplicate filename_hr in selection"

print(f"\nSelected {len(selected)} records ({SAMPLES_PER_CLASS} per class).")
print(f"Unique patients: {selected['patient_id'].nunique()}  "
      f"Unique ECG IDs: {selected['ecg_id'].nunique()}")
for name_c, lbl in [("Normal", 0), ("MI", 1), ("Abnormal", 2), ("History_MI", 3)]:
    print(f"  {name_c:<12} {(selected['label']==lbl).sum()}")

# ── Check that sample waveform files exist ────────────────────
sample_stem = PTBXL_ROOT / selected.iloc[0]["filename_hr"]
if not Path(str(sample_stem) + ".dat").exists():
    sys.exit(
        f"\nERROR: PTB-XL waveform files not found for:\n  {sample_stem}.dat\n"
        "Make sure the records500/ subfolder is present inside PTBXL_ROOT."
    )

# ── Create output directories ─────────────────────────────────
for name in CLASS_NAMES:
    (OUTPUT_DIR / name).mkdir(parents=True, exist_ok=True)

print(f"\nGenerating images into: {OUTPUT_DIR}")
print(f"Manifest will be saved: {MANIFEST}")
print("Progress every 25 records...\n")

manifest_rows = []
failed = 0

for pos, (_, row) in enumerate(selected.iterrows()):
    label       = int(row["label"])
    class_name  = CLASS_NAMES[label]
    record_stem = PTBXL_ROOT / row["filename_hr"]
    record_base = record_stem.name
    out_subdir  = OUTPUT_DIR / class_name

    # Snapshot mtime of all existing PNGs before this call
    pngs_before: dict = {
        p.name: p.stat().st_mtime
        for p in out_subdir.glob("*.png")
    }

    with tempfile.TemporaryDirectory() as _tmpdir:
        tmpdir = Path(_tmpdir)
        copied = 0
        for ext in (".dat", ".hea"):
            src = Path(str(record_stem) + ext)
            if src.exists():
                shutil.copy2(src, tmpdir / (record_base + ext))
                copied += 1

        if copied < 2:
            failed += 1
            if failed <= 5:
                print(f"  [MISSING FILES] {record_base} — {copied}/2 files found")
            continue

        cmd = [
            sys.executable, str(GEN_ECG_SCRIPT),
            "-i",  str(tmpdir),
            "-o",  str(out_subdir),
            "--max_num_images", "1",
            "-se", str(SEED),
            "-ph",
        ]

        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            # Accept only PNGs that match this record AND are new or updated
            base_short = record_base.replace("_hr", "")
            match: list = []
            for p in out_subdir.glob("*.png"):
                if base_short not in p.name:
                    continue
                before_mtime = pngs_before.get(p.name)
                if before_mtime is None:
                    match.append(p)
                    continue
                try:
                    if p.stat().st_mtime > before_mtime:
                        match.append(p)
                except OSError:
                    pass

            if match:
                img_path = sorted(match, key=lambda p: p.name)[-1]
                manifest_rows.append({
                    "image_path":          str(img_path.relative_to(OUTPUT_DIR)),
                    "label":               label,
                    "class_name":          class_name,
                    "record":              record_base,
                    "ecg_id":              int(row["ecg_id"]),
                    "patient_id":          int(row["patient_id"]),
                    "strat_fold":          int(row["strat_fold"]),
                    "filename_hr":         str(row["filename_hr"]),
                    "scp_codes":           str(row["scp_codes"]),
                    "infarction_stadium1": str(row["infarction_stadium1"]),
                    "infarction_stadium2": str(row["infarction_stadium2"]),
                })
            else:
                failed += 1
                if failed <= 5:
                    print(f"  [NO PNG]  {record_base}")
                    if r.stderr:
                        print(f"    stderr: {r.stderr[-300:]}")

        except subprocess.TimeoutExpired:
            failed += 1
            if failed <= 5:
                print(f"  [TIMEOUT] {record_base}")
        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"  [ERROR]   {record_base}: {e}")

    if (pos + 1) % 25 == 0:
        print(f"  {pos+1:>3}/{len(selected)} | "
              f"Generated: {len(manifest_rows):>3} | Failed: {failed:>3}")

print(f"\n{'='*50}")
print(f"  Generated : {len(manifest_rows)}")
print(f"  Failed    : {failed}")
print(f"{'='*50}")

if not manifest_rows:
    sys.exit("\nERROR: Nothing generated. Check error messages above.")

manifest_df = pd.DataFrame(manifest_rows)

# Final uniqueness checks on generated manifest
assert manifest_df["image_path"].nunique() == len(manifest_df), \
    "Duplicate image paths in manifest"
assert manifest_df["patient_id"].nunique() == len(manifest_df), \
    "Duplicate patient_ids in manifest"

manifest_df.to_csv(MANIFEST, index=False)
print(f"\nManifest saved: {MANIFEST}")
print(f"Columns: {list(manifest_df.columns)}")
print("\nClass breakdown:")
for lbl, name in enumerate(CLASS_NAMES):
    n = (manifest_df["label"] == lbl).sum()
    print(f"  {name:<12} {n:>3}")
print(f"\nFold distribution:")
print(manifest_df["strat_fold"].value_counts().sort_index().to_string())
print("\nNext: run ecg_domain_shift_full_pipeline.ipynb from top to bottom.")
