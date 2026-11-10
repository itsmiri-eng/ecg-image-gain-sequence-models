import numpy as np
from PIL import Image


def preprocess_image(path, settings):
    """Fixed geometry only: no labels, cohort statistics, or learned transformations."""
    with Image.open(path) as im:
        im = im.convert("L")
        w, h = im.size
        a, b, c, d = settings["crop_fraction_xyxy"]
        box = (int(a * w), int(b * h), int(c * w), int(d * h))
        im = im.crop(box).resize((settings["width"], settings["height"]), Image.Resampling.BILINEAR)
        return (1.0 - np.asarray(im, dtype=np.float32) / 255.0)[None, :, :]
