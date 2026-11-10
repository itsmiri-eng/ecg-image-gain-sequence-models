"""Synthetic physical-geometry and image-integrity tests; no dataset rendering."""
import copy
import hashlib
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import src.ptbxl_rendering as rendering


CONFIG_PATH = Path(__file__).resolve().parents[1] / "experiments/configs/ptbxl_images_v1.json"


class RenderingTest(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(CONFIG_PATH.read_text())
        self.n = self.config["samples_per_lead"]
        self.zero = np.zeros((self.n, 12), dtype=np.float64)

    def assert_row_and_page_bounds(self, points, meta):
        self.assertEqual(len(points), 12)
        for point, panel in zip(points, meta["lead_panels"]):
            self.assertEqual(point.shape, (self.n, 2))
            self.assertEqual(point.dtype, np.int32)
            self.assertTrue(np.all(point[:, 0] >= 0))
            self.assertTrue(np.all(point[:, 0] < meta["width"]))
            self.assertTrue(np.all(point[:, 1] >= panel["row_top_y_pixels"]))
            self.assertTrue(np.all(point[:, 1] < panel["row_bottom_y_pixels"]))
            self.assertTrue(np.all(point[:, 1] < meta["height"]))

    def test_common_time_axis_has_all_source_samples_and_correct_duration(self):
        points, meta = rendering.geometry(self.zero, self.config)
        self.assertEqual(self.n, 5000)
        self.assertEqual(meta["source_points_plotted"], 60000)
        self.assertEqual(meta["duration_seconds"], 10)
        self.assertAlmostEqual(meta["last_sample_time_seconds"], 9.998)
        self.assertEqual([p["lead"] for p in meta["lead_panels"]], list(rendering.LEADS))
        expected_x = np.rint((self.config["left_margin_mm"] + np.arange(self.n) /
                              self.config["sampling_rate_hz"] * self.config["speed_mm_per_second"])
                             * self.config["pixels_per_mm"]).astype(np.int32)
        for point, panel in zip(points, meta["lead_panels"]):
            np.testing.assert_array_equal(point[:, 0], expected_x)
            self.assertEqual(panel["sample_start"], 0)
            self.assertEqual(panel["sample_stop_exclusive"], 5000)
            self.assertEqual(panel["time_stop_exclusive_seconds"], 10)
        self.assertAlmostEqual(meta["seconds_per_pixel"],
                               1 / (self.config["pixels_per_mm"] * self.config["speed_mm_per_second"]))
        self.assert_row_and_page_bounds(points, meta)

    def test_positive_voltage_goes_up_negative_down_and_dc_offset_is_not_removed(self):
        signal = self.zero.copy()
        signal[0, :] = 1.0
        signal[1, :] = -1.0
        signal[:, 2] = 0.5
        original = signal.copy()
        points, meta = rendering.geometry(signal, self.config)
        self.assertEqual(meta["gain_mm_per_mv"], self.config["nominal_gain_mm_per_mv"])
        pixels_per_mv = meta["gain_mm_per_mv"] * self.config["pixels_per_mm"]
        for column, point in enumerate(points):
            baseline = meta["lead_panels"][column]["baseline_y_pixels"]
            expected_y = baseline - original[:, column] * pixels_per_mv
            np.testing.assert_allclose(point[:, 1], expected_y, atol=0.5, rtol=0)
        self.assertLess(points[0][0, 1], meta["lead_panels"][0]["baseline_y_pixels"])
        self.assertGreater(points[0][1, 1], meta["lead_panels"][0]["baseline_y_pixels"])
        np.testing.assert_array_equal(signal, original)
        self.assert_row_and_page_bounds(points, meta)

    def test_32mv_outlier_reduces_one_common_gain_without_dropping_or_clamping_samples(self):
        signal = self.zero.copy()
        signal[123, 11], signal[456, 4] = 32.0, -32.0
        signal[789, :] = 1.0
        points, meta = rendering.geometry(signal, self.config)
        self.assertEqual(meta["gain_reduction_steps"], 5)
        self.assertEqual(meta["gain_mm_per_mv"], 0.3125)
        scale = meta["gain_mm_per_mv"] * self.config["pixels_per_mm"]
        for column, point in enumerate(points):
            baseline = meta["lead_panels"][column]["baseline_y_pixels"]
            np.testing.assert_allclose(point[:, 1], baseline - signal[:, column] * scale, atol=.5, rtol=0)
        self.assertEqual(meta["source_points_plotted"], signal.size)
        self.assertEqual(meta["out_of_row_point_count"], 0)
        self.assertEqual(meta["clipped_point_count"], 0)
        self.assert_row_and_page_bounds(points, meta)

    def test_zero_signal_retains_all_flat_leads_at_nominal_gain(self):
        image, meta = rendering.render_page(self.zero, self.config)
        self.assertEqual(image.dtype, np.uint8)
        self.assertEqual(image.shape, (meta["height"], meta["width"]))
        self.assertEqual(meta["flat_leads"], list(rendering.LEADS))
        self.assertEqual(meta["gain_mm_per_mv"], 10)
        self.assertIn("flat_lead", meta["quality_flags"])
        self.assertFalse(meta["low_gain_flag"])
        self.assertFalse(meta["printed_identifiers_or_diagnoses"])
        self.assertGreater(int(image.max()), int(image.min()))

    def test_calibration_pulse_has_same_voltage_and_time_scale_as_trace(self):
        collected = []
        real_polyline = rendering.cv2.polylines

        def capture(image, arrays, *args, **kwargs):
            collected.extend(np.array(a, copy=True) for a in arrays)
            return real_polyline(image, arrays, *args, **kwargs)

        with patch.object(rendering.cv2, "polylines", side_effect=capture):
            _, meta = rendering.render_page(self.zero, self.config)
        waveforms = [array for array in collected if len(array) == 5000]
        pulses = [array for array in collected if len(array) == 6]
        self.assertEqual(len(waveforms), 12)
        self.assertEqual(len(pulses), 12)
        expected_height = 1 / meta["mv_per_pixel"]
        expected_width = .2 / meta["seconds_per_pixel"]
        self.assertAlmostEqual(meta["calibration_pulse_height_pixels"], expected_height)
        self.assertAlmostEqual(meta["calibration_pulse_width_pixels"], expected_width)
        for pulse, panel in zip(pulses, meta["lead_panels"]):
            self.assertAlmostEqual(pulse[1, 1] - pulse[2, 1], expected_height, delta=.5)
            self.assertAlmostEqual(pulse[3, 0] - pulse[2, 0], expected_width, delta=.5)
            self.assertEqual(pulse[0, 1], round(panel["baseline_y_pixels"]))
            self.assertLess(pulse[:, 0].max(), meta["signal_x_start_pixels"])
            self.assertGreaterEqual(pulse[:, 1].min(), panel["row_top_y_pixels"])
        self.assertEqual(meta["calibration_pulse_raster_height_pixels"], int(pulses[-1][1, 1] - pulses[-1][2, 1]))

    def test_very_low_gain_reports_quantized_calibration_and_quality_warning(self):
        signal = self.zero.copy()
        signal[100, 0] = 200.0
        _, meta = rendering.render_page(signal, self.config)
        self.assertTrue(meta["low_gain_flag"])
        self.assertIn("gain_below_2.5_mm_per_mv", meta["quality_flags"])
        self.assertIn("calibration_pulse_below_2_pixels", meta["quality_flags"])
        self.assertLess(meta["calibration_pulse_raster_height_pixels"], 2)
        self.assertAlmostEqual(meta["calibration_pulse_raster_height_pixels"],
                               meta["calibration_pulse_height_pixels"], delta=.5)

    def test_text_and_pulse_metadata_follow_accepted_nondefault_time_configuration(self):
        config = copy.deepcopy(self.config)
        config.update(samples_per_lead=2500, speed_mm_per_second=20)
        texts = []
        with patch.object(rendering.cv2, "putText", side_effect=lambda image, text, *a, **k: texts.append(text) or image):
            _, meta = rendering.render_page(np.zeros((2500, 12)), config)
        self.assertIn("5 s", texts[0])
        self.assertIn("20 mm/s", texts[0])
        self.assertIn("0.000-4.998 s", texts[-1])
        self.assertIn("500 Hz source", texts[-1])
        self.assertEqual(meta["calibration_pulse_width_pixels"], .2 * 20 * config["pixels_per_mm"])

    def test_renderer_has_no_identity_argument_and_sensitive_config_values_are_not_printed(self):
        self.assertEqual(list(inspect.signature(rendering.render_page).parameters), ["signal", "config"])
        config = copy.deepcopy(self.config)
        config.update(patient_id="SECRET_PATIENT", ecg_id="SECRET_RECORD", diagnosis="SECRET_DIAGNOSIS")
        texts = []
        with patch.object(rendering.cv2, "putText", side_effect=lambda image, text, *a, **k: texts.append(text) or image):
            rendering.render_page(self.zero, config)
        self.assertEqual(len(texts), 14)
        self.assertEqual(texts[1:-1], list(rendering.DISPLAY_LEADS))
        self.assertNotIn("SECRET", " ".join(texts))
        for diagnostic_name in config["class_names"]:
            self.assertNotIn(diagnostic_name, " ".join(texts))
        with self.assertRaises(TypeError):
            rendering.render_page(self.zero, config, patient_id="not accepted")

    def test_png_is_deterministic_lossless_and_save_refuses_overwrite(self):
        time = np.arange(self.n) / self.config["sampling_rate_hz"]
        signal = np.tile(.2 * np.sin(2*np.pi*time)[:, None], (1, 12))
        image, _ = rendering.render_page(signal, self.config)
        payload = rendering.encode_png(image, self.config)
        self.assertEqual(payload, rendering.encode_png(image, self.config))
        repeated, _ = rendering.render_page(signal, self.config)
        np.testing.assert_array_equal(image, repeated)
        self.assertEqual(payload[:8], b"\x89PNG\r\n\x1a\n")
        decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        np.testing.assert_array_equal(image, decoded)
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "synthetic.png"
            meta = rendering.save_page(signal, self.config, destination)
            saved = destination.read_bytes()
            self.assertEqual(saved, payload)
            self.assertEqual(meta["image_sha256"], hashlib.sha256(saved).hexdigest())
            self.assertEqual(meta["pixel_sha256"], hashlib.sha256(image.tobytes()).hexdigest())
            with self.assertRaises(FileExistsError):
                rendering.save_page(self.zero, self.config, destination)
            self.assertEqual(destination.read_bytes(), saved)

    def test_invalid_shapes_nonfinite_values_geometry_and_encoding_fail(self):
        invalids = [np.zeros((4999, 12)), np.zeros((5000, 11)), np.zeros((12, 5000)),
                    np.full((5000, 12), np.nan), np.full((5000, 12), np.inf)]
        for invalid in invalids:
            with self.subTest(shape=invalid.shape), self.assertRaises(ValueError):
                rendering.geometry(invalid, self.config)
        for changes in [{"lead_order": list(reversed(rendering.LEADS))}, {"pixels_per_mm": 0},
                        {"row_signal_half_height_mm": self.config["row_pitch_mm"]}]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                rendering.geometry(self.zero, {**self.config, **changes})
        for invalid in [np.zeros((4, 4), np.float32), np.zeros((4, 4, 3), np.uint8)]:
            with self.assertRaises(ValueError):
                rendering.encode_png(invalid, self.config)

    def test_waveform_loader_reorders_leads_without_altering_calibrated_values(self):
        order = list(reversed(rendering.LEADS))
        signal = np.tile(np.array([rendering.LEADS.index(name) for name in order]), (self.n, 1)).astype(float)
        fields = {"sig_name": order, "fs": 500, "units": ["mV"]*12}
        with patch.object(rendering.wfdb, "rdsamp", return_value=(signal, fields)) as read:
            result, meta = rendering.load_waveform("not-a-real-record", self.config)
        read.assert_called_once_with("not-a-real-record")
        np.testing.assert_array_equal(result, np.tile(np.arange(12), (self.n, 1)))
        self.assertEqual(meta["source_lead_order"], order)
        self.assertEqual(meta["rendered_lead_order"], list(rendering.LEADS))
        self.assertTrue(result.flags.c_contiguous)

    def test_waveform_loader_rejects_duplicate_unknown_leads_units_rate_and_nonfinite(self):
        good = {"sig_name": list(rendering.LEADS), "fs": 500, "units": ["mV"]*12}
        variants = [{**good, "sig_name": ["I"]*12},
                    {**good, "sig_name": [*rendering.LEADS[:-1], "UNKNOWN"]},
                    {**good, "fs": 100}, {**good, "units": ["uV"]*12}]
        for fields in variants:
            with self.subTest(fields=fields), patch.object(rendering.wfdb, "rdsamp", return_value=(self.zero, fields)):
                with self.assertRaises(ValueError):
                    rendering.load_waveform("not-a-real-record", self.config)
        for signal in [np.full_like(self.zero, np.nan), np.zeros((4999, 12))]:
            with patch.object(rendering.wfdb, "rdsamp", return_value=(signal, good)):
                with self.assertRaises(ValueError):
                    rendering.load_waveform("not-a-real-record", self.config)


if __name__ == "__main__":
    unittest.main()
