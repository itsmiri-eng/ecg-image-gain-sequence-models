"""Raw 100 Hz PTB-XL waveforms for the signal-domain reference model (protocol ptbxl_v2_signal_resnet1d).

Usage (builds the cache once; later calls only verify it):
    python -X utf8 -m src.ptbxl_signals

The cohort, labels, patient IDs and row order are the v1 image cache's rows for each split, so every
signal model sees exactly the ECGs the image models saw, in the same order. Each records100 .dat/.hea
file is checked against PTB-XL's SHA256SUMS.txt before it is read. Signals are stored unfiltered in mV
as float32 [N, 12, 1000] per split; per-lead normalization is fitted on training signals only.
"""
from __future__ import annotations

import json

import numpy as np
import torch
from torch.utils.data import Dataset

from src.common import ROOT, read_json, sha256_file, utc_now, write_json
from src.ptbxl_learning_data import CLASS_NAMES

SOURCE = ROOT / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
CACHE = ROOT / "data/processed/ptbxl_signals100_v1"
IMAGE_CONFIG = ROOT / "experiments/configs/ptbxl_v2_cnn_pool.json"
SHAPE = (12, 1000)
LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
SPLITS = ("train", "validation", "test")


def image_rows(split):
    """The v1 image cache's rows for `split` (cohort, labels and order shared with every image model)."""
    from src.run_ptbxl_experiment import make_dataset
    return make_dataset(read_json(IMAGE_CONFIG), split, normalize=False).rows


def _published_hashes():
    hashes = {}
    for line in (SOURCE / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        hashes[name.strip()] = digest
    return hashes


def _read(row, published):
    import wfdb
    for suffix in (".dat", ".hea"):
        name = row["filename_lr"] + suffix
        if sha256_file(SOURCE / name) != published[name]:
            raise ValueError(f"PTB-XL file differs from SHA256SUMS.txt: {name}")
    signal, fields = wfdb.rdsamp(str(SOURCE / row["filename_lr"]))
    if fields["fs"] != 100 or list(fields["sig_name"]) != LEADS or signal.shape != SHAPE[::-1]:
        raise ValueError(f"Unexpected record layout: {row['filename_lr']}")
    if not np.isfinite(signal).all():
        raise ValueError(f"Nonfinite samples: {row['filename_lr']}")
    return signal.T.astype(np.float32)


def build():
    """Write per-split signal arrays, their ECG IDs and a metadata file with hashes and training normalization."""
    if (CACHE / "metadata.json").exists():
        raise FileExistsError(f"{CACHE} already built; never rebuilt in place")
    CACHE.mkdir(parents=True, exist_ok=True)
    published, files = _published_hashes(), {}
    for split in SPLITS:
        rows = image_rows(split)
        signals = np.stack([_read(row, published) for row in rows])
        ids = np.asarray([int(row["ecg_id"]) for row in rows])
        np.save(CACHE / f"{split}.npy", signals)
        np.save(CACHE / f"{split}_ecg_ids.npy", ids)
        files[split] = {"records": len(rows), "signals_sha256": sha256_file(CACHE / f"{split}.npy"),
                        "ecg_ids_sha256": sha256_file(CACHE / f"{split}_ecg_ids.npy")}
        print(f"{utc_now()} {split}: {signals.shape}", flush=True)
        if split == "train":
            normalization = {"fit_partition": "train", "training_records": len(rows),
                             "mean": signals.mean(axis=(0, 2), dtype=np.float64).tolist(),
                             "std": signals.std(axis=(0, 2), dtype=np.float64).tolist()}
    write_json(CACHE / "metadata.json", {"status": "complete", "created_utc": utc_now(), "shape": list(SHAPE),
               "leads": LEADS, "sampling_rate_hz": 100, "units": "mV", "source": "PTB-XL v1.0.3 records100",
               "source_check": "every .dat/.hea SHA-256 matched SHA256SUMS.txt", "files": files,
               "normalization": normalization, "builder_sha256": sha256_file(ROOT / "src/ptbxl_signals.py")})


class SignalDataset(Dataset):
    """(signal [12, 1000] float32, targets [5] float32) in image-cache order; hash-checked when opened."""
    def __init__(self, split, normalize=True, cache_root=CACHE):
        if split not in SPLITS:
            raise ValueError("split must be train, validation or test")
        self.metadata = read_json(cache_root / "metadata.json")
        info = self.metadata["files"][split]
        if (self.metadata.get("status") != "complete" or sha256_file(cache_root / f"{split}.npy") != info["signals_sha256"]
                or sha256_file(cache_root / f"{split}_ecg_ids.npy") != info["ecg_ids_sha256"]):
            raise ValueError("Signal cache incomplete or changed")
        self.rows, self.split, self.normalize = image_rows(split), split, bool(normalize)
        ids = np.load(cache_root / f"{split}_ecg_ids.npy")
        if ids.tolist() != [int(row["ecg_id"]) for row in self.rows]:
            raise ValueError("Signal cache order differs from the image cache rows")
        self.signals = np.load(cache_root / f"{split}.npy")
        self.normalization = self.metadata["normalization"]
        self.mean = np.asarray(self.normalization["mean"], np.float32)[:, None]
        self.std = np.asarray(self.normalization["std"], np.float32)[:, None]
        self.targets = np.asarray([[row[f"label_{n}"] for n in CLASS_NAMES] for row in self.rows], np.float32)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        x = self.signals[index]
        if self.normalize:
            x = (x - self.mean) / self.std
        return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)), torch.from_numpy(self.targets[index])


if __name__ == "__main__":
    if not (CACHE / "metadata.json").exists():
        build()
    for split in SPLITS:
        dataset = SignalDataset(split)
        print(json.dumps({"split": split, "records": len(dataset), "first_ecg": dataset.rows[0]["ecg_id"]}))
