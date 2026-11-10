"""Conservative image-similarity screening before model fitting; never deletes images.

Perceptual hashes screen candidate pairs. Extremely similar crop tensors are
grouped to avoid crossing split boundaries. This does not establish that every
remaining pair comes from a different patient or acquisition.
"""
import hashlib
from collections import Counter

import numpy as np
from scipy.fft import dctn
from PIL import Image

from src.common import ROOT, read_json, sha256_file, write_csv, write_json
from src.preprocessing.images import preprocess_image


def main():
    cfg = read_json(ROOT / "experiments/configs/baseline_v1.json")["preprocessing"]
    seen, records, hashes, images = set(), [], [], []
    for p in sorted((ROOT / "ecg files").glob("*/*.jpg")):
        digest = sha256_file(p)
        if digest in seen:
            continue
        seen.add(digest)
        x = preprocess_image(p, cfg)[0]
        small = np.asarray(Image.fromarray(x).resize((32, 32), Image.Resampling.BILINEAR))
        frequency = dctn(small, norm="ortho")[:8, :8].flatten()[1:]
        hashes.append(frequency > np.median(frequency))
        records.append({"path": str(p.relative_to(ROOT)), "file_sha256": digest})
        images.append(x)
    hashes, images = np.array(hashes), np.array(images)
    candidates, links = [], []
    for a in range(len(records)):
        for b in range(a + 1, len(records)):
            distance = int(np.count_nonzero(hashes[a] != hashes[b]))
            if distance > 6:
                continue
            xa, xb = images[a].ravel().astype(float), images[b].ravel().astype(float)
            corr = float(np.corrcoef(xa, xb)[0, 1])
            mae = float(np.abs(xa - xb).mean())
            linked = corr >= .995 and mae <= .005
            row = {"path_a": records[a]["path"], "path_b": records[b]["path"],
                   "sha256_a": records[a]["file_sha256"], "sha256_b": records[b]["file_sha256"],
                   "phash_hamming": distance, "crop_correlation": corr, "mean_absolute_ink_difference": mae,
                   "grouping_link": linked}
            candidates.append(row)
            if linked:
                links.append({"sha256_a": row["sha256_a"], "sha256_b": row["sha256_b"],
                              "reason": "phash<=6_and_crop_correlation>=0.995_and_MAE<=0.005"})
    write_csv(ROOT / "data/audit/near_duplicate_candidates.csv", candidates,
              ["path_a", "path_b", "sha256_a", "sha256_b", "phash_hamming", "crop_correlation", "mean_absolute_ink_difference", "grouping_link"])
    write_csv(ROOT / "data/audit/near_duplicate_links.csv", links, ["sha256_a", "sha256_b", "reason"])
    summary = {"unique_byte_images_screened": len(records), "candidate_pairs": len(candidates),
               "conservative_grouping_links": len(links),
               "method": "63-bit non-DC DCT pHash distance<=6; link if crop Pearson r>=0.995 and mean absolute ink difference<=0.005",
               "preprocessing": cfg, "model_results_used": False,
               "limitation": "Similarity screen can miss shifted/rescaled/redrawn ECGs; no assertion of exhaustive near-duplicate removal or patient independence"}
    write_json(ROOT / "data/audit/near_duplicate_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
