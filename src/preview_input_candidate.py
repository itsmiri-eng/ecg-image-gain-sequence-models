"""Illustrate one training-only crop candidate; never replaces an experiment input."""
import numpy as np
from PIL import Image, ImageOps

from src.common import ROOT, read_json, sha256_file, write_json
from src.verify import load_verified_data


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = read_json(ROOT / "experiments/configs/crop_candidate_dev_only.json")
    x, _, rows, _, _ = load_verified_data()
    row = next(r for r in rows if r["sample_id"] == cfg["sample_id"])
    if row["split"] != cfg["required_partition"] or row["split"] == "test":
        raise ValueError("Preview case must be the declared development case")
    with Image.open(ROOT / row["path"]) as im:
        crop = im.convert("L").crop(tuple(cfg["crop_xyxy_pixels"]))
    target = (cfg["target_width"], cfg["target_height"])
    fitted = ImageOps.contain(crop, target, method=Image.Resampling.BILINEAR)
    preview = Image.new("L", target, 255)
    preview.paste(fitted, ((target[0] - fitted.width) // 2, (target[1] - fitted.height) // 2))
    out = ROOT / "results/readiness/inputs"
    out.mkdir(parents=True, exist_ok=True)
    preview.save(out / "candidate_hb59_image.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].imshow(1.0 - x[int(row["array_index"]), 0], cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Frozen baseline: 256×128\nTop peaks clipped; existing results preserved")
    axes[1].imshow(np.asarray(preview), cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Candidate preview: 512×320 with padding\nTaller margin retains known peaks; header text remains")
    for axis in axes:
        axis.axis("off")
    fig.suptitle("One training case: crop geometry review, not a model comparison")
    fig.savefig(out / "candidate_hb59_comparison.png", dpi=150)
    plt.close(fig)
    write_json(out / "candidate_preview.json", {
        "config": cfg, "sample_id": row["sample_id"], "source_path": row["path"], "partition": row["split"],
        "source_sha256": row["file_sha256"], "baseline_input_sha256": row["input_sha256"],
        "native_crop_size": list(crop.size), "fitted_size": list(fitted.size),
        "preview_sha256": sha256_file(out / "candidate_hb59_image.png"),
        "script_sha256": sha256_file(__file__),
        "adopted_for_training": False, "new_model_inference": False, "test_images_used": False,
    })
    print("Training-only candidate preview saved; frozen inputs and baselines unchanged.")


if __name__ == "__main__":
    main()
