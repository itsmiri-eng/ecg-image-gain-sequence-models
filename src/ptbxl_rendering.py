"""Deterministic, calibrated clean pages from real PTB-XL waveforms.

No diagnoses, demographics or identifiers are accepted by the pixel renderer.
Every lead is plotted over the full source duration, with shared horizontal time.
"""
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
import wfdb

LEADS = ("I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6")
DISPLAY_LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")


def load_waveform(record_path, config):
    signal, fields = wfdb.rdsamp(str(record_path))
    names = [str(s).upper() for s in fields["sig_name"]]
    if len(names) != len(set(names)) or set(names) != set(LEADS):
        raise ValueError("Expected exactly the 12 standard lead names")
    if fields["fs"] != config["sampling_rate_hz"]:
        raise ValueError("Sampling frequency differs from the frozen protocol")
    if any(str(unit).lower() != "mv" for unit in fields["units"]):
        raise ValueError("Expected WFDB-calibrated millivolt units")
    order = [names.index(name) for name in LEADS]
    signal = np.ascontiguousarray(signal[:, order], dtype=np.float64)
    if signal.shape != (config["samples_per_lead"], 12) or not np.isfinite(signal).all():
        raise ValueError("Invalid waveform dimensions or nonfinite signal")
    return signal, {"sampling_rate_hz": float(fields["fs"]), "units": "mV",
                    "source_lead_order": names, "rendered_lead_order": list(LEADS),
                    "samples_per_lead": len(signal)}


def geometry(signal, config):
    signal = np.asarray(signal, dtype=np.float64)
    if signal.shape != (config["samples_per_lead"], 12) or not np.isfinite(signal).all():
        raise ValueError("Expected finite full-length 12-lead waveforms")
    if tuple(config["lead_order"]) != LEADS:
        raise ValueError("Unsupported render lead order")
    ppm = config["pixels_per_mm"]
    pitch, half = config["row_pitch_mm"], config["row_signal_half_height_mm"]
    if ppm <= 0 or not 0 < half < pitch / 2:
        raise ValueError("Invalid calibrated row geometry")
    gain = float(config["nominal_gain_mm_per_mv"])
    maximum = float(np.abs(signal).max())
    reductions = max(0, math.ceil(math.log2(maximum * gain / half))) if maximum else 0
    gain /= 2 ** reductions
    duration = len(signal) / config["sampling_rate_hz"]
    width_mm = config["left_margin_mm"] + duration * config["speed_mm_per_second"] + config["right_margin_mm"]
    height_mm = config["header_mm"] + 12 * pitch + config["footer_mm"]
    width, height = round(width_mm * ppm), round(height_mm * ppm)
    x = np.rint((config["left_margin_mm"] + np.arange(len(signal)) / config["sampling_rate_hz"]
                 * config["speed_mm_per_second"]) * ppm).astype(np.int32)
    baselines = (config["header_mm"] + (np.arange(12) + 0.5) * pitch) * ppm
    points = []
    bounds = []
    out_of_row = 0
    for i in range(12):
        y_float = baselines[i] - signal[:, i] * gain * ppm
        y = np.rint(y_float).astype(np.int32)
        row_top = (config["header_mm"] + i * pitch) * ppm
        row_bottom = row_top + pitch * ppm
        out_of_row += int(np.count_nonzero((y_float < row_top) | (y_float >= row_bottom)))
        points.append(np.column_stack((x, y)))
        bounds.append({"lead": LEADS[i], "sample_start": 0, "sample_stop_exclusive": len(signal),
                       "time_start_seconds": 0, "time_stop_exclusive_seconds": duration,
                       "baseline_y_pixels": float(baselines[i]), "row_top_y_pixels": float(row_top),
                       "row_bottom_y_pixels": float(row_bottom), "trace_min_y_pixels": int(y.min()),
                       "trace_max_y_pixels": int(y.max())})
    if out_of_row or any((p[:, 0] < 0).any() or (p[:, 0] >= width).any() or (p[:, 1] < 0).any()
                         or (p[:, 1] >= height).any() for p in points):
        raise ValueError("Waveform would extend outside its row/page")
    return points, {"width": width, "height": height, "pixels_per_mm": ppm,
                    "speed_mm_per_second": config["speed_mm_per_second"], "gain_mm_per_mv": gain,
                    "gain_reduction_steps": reductions, "seconds_per_pixel": 1/(ppm*config["speed_mm_per_second"]),
                    "mv_per_pixel": 1/(ppm*gain), "maximum_absolute_mv": maximum,
                    "duration_seconds": duration, "last_sample_time_seconds": (len(signal)-1)/config["sampling_rate_hz"],
                    "signal_x_start_pixels": int(x[0]), "signal_x_last_pixels": int(x[-1]),
                    "source_points_plotted": int(signal.size), "out_of_row_point_count": out_of_row,
                    "clipped_point_count": 0, "nonfinite_point_count": 0,
                    "flat_leads": [LEADS[i] for i in range(12) if np.ptp(signal[:, i]) == 0],
                    "lead_panels": bounds}


@lru_cache(maxsize=4)
def _background(config_json, width, height):
    cfg = json.loads(config_json)
    img = np.full((height, width), cfg["background_gray"], dtype=np.uint8)
    ppm = cfg["pixels_per_mm"]
    top = round(cfg["header_mm"] * ppm)
    bottom = round((cfg["header_mm"] + 12*cfg["row_pitch_mm"]) * ppm)
    # Grid spans the lead rows only; one-mm and five-mm spacing is isotropic.
    for millimetre in range(int(width/ppm) + 1):
        x = round(millimetre*ppm)
        if x < width:
            img[top:bottom, x] = cfg["major_grid_gray"] if millimetre % 5 == 0 else cfg["minor_grid_gray"]
    for millimetre in range(12*cfg["row_pitch_mm"] + 1):
        y = top + round(millimetre*ppm)
        if y < height:
            img[y, :] = cfg["major_grid_gray"] if millimetre % 5 == 0 else cfg["minor_grid_gray"]
    return img


def render_page(signal, config):
    """Return uint8 grayscale page + geometric QA; accepts no labels/identifiers."""
    points, meta = geometry(signal, config)
    canonical = json.dumps(config, sort_keys=True)
    image = _background(canonical, meta["width"], meta["height"]).copy()
    ppm, gain = config["pixels_per_mm"], meta["gain_mm_per_mv"]
    ink = config["trace_gray"]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = ppm / 4
    header = f"12-lead ECG | {meta['duration_seconds']:g} s | {config['speed_mm_per_second']:g} mm/s | {gain:g} mm/mV | 1 mV calibration"
    cv2.putText(image, header, (round(2*ppm), round(6*ppm)), font, 0.45*font_scale, ink, 1, cv2.LINE_AA)
    pulse_width_mm = 0.2 * config["speed_mm_per_second"]
    for i, point_array in enumerate(points):
        cv2.polylines(image, [point_array], False, ink, config["line_width_pixels"], cv2.LINE_AA)
        panel = meta["lead_panels"][i]
        label_y = round(panel["row_top_y_pixels"] + 4*ppm)
        cv2.putText(image, DISPLAY_LEADS[i], (round(2*ppm), label_y), font, 0.40*font_scale, ink, 1, cv2.LINE_AA)
        base = panel["baseline_y_pixels"]
        # A 1-mV, 0.2-second pulse at the same physical scale as the waveform.
        pulse = np.rint([(12*ppm, base), (14*ppm, base), (14*ppm, base-gain*ppm),
                         ((14+pulse_width_mm)*ppm, base-gain*ppm), ((14+pulse_width_mm)*ppm, base),
                         ((16+pulse_width_mm)*ppm, base)]).astype(np.int32)
        cv2.polylines(image, [pulse], False, ink, 1, cv2.LINE_AA)
    footer = f"All leads: 0.000-{meta['last_sample_time_seconds']:.3f} s | {config['sampling_rate_hz']:g} Hz source | Clean waveform rendering"
    cv2.putText(image, footer, (round(2*ppm), meta["height"]-round(3*ppm)), font, 0.40*font_scale, ink, 1, cv2.LINE_AA)
    meta["calibration_pulse_height_pixels"] = float(gain*ppm)
    meta["calibration_pulse_width_pixels"] = float(pulse_width_mm*ppm)
    meta["calibration_pulse_raster_height_pixels"] = int(pulse[1,1]-pulse[2,1])
    meta["calibration_pulse_raster_width_pixels"] = int(pulse[3,0]-pulse[2,0])
    meta["pixel_mode"] = "L"
    meta["printed_identifiers_or_diagnoses"] = False
    meta["low_gain_flag"] = bool(gain < 2.5)
    meta["quality_flags"] = (["gain_below_2.5_mm_per_mv"] if gain < 2.5 else []) + (["flat_lead"] if meta["flat_leads"] else []) + (["calibration_pulse_below_2_pixels"] if meta["calibration_pulse_raster_height_pixels"] < 2 else [])
    return image, meta


def encode_png(image, config):
    if image.dtype != np.uint8 or image.ndim != 2:
        raise ValueError("Expected an 8-bit grayscale page")
    ok, buffer = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, config["png_compression"]])
    if not ok:
        raise RuntimeError("PNG encoder failed")
    return buffer.tobytes()


def save_page(signal, config, destination):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError("Refusing to overwrite an existing ECG page")
    image, meta = render_page(signal, config)
    payload = encode_png(image, config)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as f:
        f.write(payload)
    return {**meta, "image_sha256": hashlib.sha256(payload).hexdigest(),
            "pixel_sha256": hashlib.sha256(image.tobytes()).hexdigest(), "image_bytes": len(payload)}
