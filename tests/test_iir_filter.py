import unittest
from unittest.mock import patch

import numpy as np
from scipy.signal import freqz

from iir_filter.design import design_iir
from iir_filter.infer import infer_iir_params


FS = 48000


def db(value):
    return 20 * np.log10(max(abs(value), np.finfo(float).tiny))


class DesignIIRTests(unittest.TestCase):
    def assert_finite_filter(self, b, a):
        self.assertTrue(np.all(np.isfinite(b)))
        self.assertTrue(np.all(np.isfinite(a)))
        self.assertAlmostEqual(a[0], 1.0)

    def test_biquad_bandpass_designs_second_order_filter(self):
        b, a = design_iir(
            {"ftype": "bandpass", "f0": 1000, "Q": 5, "order": 2, "method": "biquad"},
            fs=FS,
        )

        self.assertEqual(len(b), 3)
        self.assertEqual(len(a), 3)
        self.assert_finite_filter(b, a)

    def test_scipy_notch_maps_to_bandstop(self):
        b, a = design_iir(
            {"ftype": "notch", "f0": 1000, "Q": 10, "order": 2, "method": "butterworth"},
            fs=FS,
        )

        self.assertEqual(len(b), 5)
        self.assertEqual(len(a), 5)
        self.assert_finite_filter(b, a)

        f, h = freqz(b, a, worN=4096, fs=FS)
        center_idx = int(np.argmin(np.abs(f - 1000)))
        self.assertGreater(db(h[0]), -1)
        self.assertLess(db(h[center_idx]), -10)

    def test_q_controls_butterworth_band_edges(self):
        with patch(
            "iir_filter.design.butter",
            return_value=(np.array([1.0]), np.array([1.0])),
        ) as butter_mock:
            design_iir(
                {"ftype": "bandpass", "f0": 1000, "Q": 10, "order": 2, "method": "butterworth"},
                fs=10000,
            )

        args, kwargs = butter_mock.call_args
        np.testing.assert_allclose(args[1], [0.19, 0.21])
        self.assertEqual(kwargs["btype"], "bandpass")

    def test_q_controls_ellip_alias_band_edges(self):
        with patch(
            "iir_filter.design.ellip",
            return_value=(np.array([1.0]), np.array([1.0])),
        ) as ellip_mock:
            design_iir(
                {
                    "ftype": "notch",
                    "f0": 1000,
                    "Q": 10,
                    "order": 2,
                    "method": "ellip",
                    "rp": 1,
                    "rs": 40,
                },
                fs=10000,
            )

        args, kwargs = ellip_mock.call_args
        np.testing.assert_allclose(args[3], [0.19, 0.21])
        self.assertEqual(kwargs["btype"], "bandstop")


class InferIIRTests(unittest.TestCase):
    def test_infer_biquad_bandpass_returns_complete_metadata(self):
        b, a = design_iir(
            {"ftype": "bandpass", "f0": 1000, "Q": 5, "order": 2, "method": "biquad"},
            fs=FS,
        )

        inferred = infer_iir_params(b, a, FS)

        self.assertEqual(inferred["ftype"], "bandpass")
        self.assertAlmostEqual(inferred["f0"], 1000, delta=10)
        self.assertIsNotNone(inferred["Q"])
        self.assertGreater(inferred["Q"], 0)
        self.assertEqual(inferred["order"], 2)
        self.assertEqual(inferred["fs"], float(FS))
        self.assertIn("designable", inferred)

    def test_infer_biquad_notch_does_not_crash(self):
        b, a = design_iir(
            {"ftype": "notch", "f0": 1000, "Q": 5, "order": 2, "method": "biquad"},
            fs=FS,
        )

        inferred = infer_iir_params(b, a, FS)

        self.assertEqual(inferred["ftype"], "notch")
        self.assertAlmostEqual(inferred["f0"], 1000, delta=10)
        self.assertIsNotNone(inferred["Q"])
        self.assertGreater(inferred["Q"], 0)
        self.assertEqual(inferred["order"], 2)

    def test_inferred_result_is_guarded_before_redesign(self):
        b, a = design_iir(
            {"ftype": "bandpass", "f0": 1000, "Q": 5, "order": 2, "method": "biquad"},
            fs=FS,
        )
        inferred = infer_iir_params(b, a, FS)

        if inferred["designable"]:
            b2, a2 = design_iir(inferred)
            self.assert_finite_coefficients(b2, a2)
        else:
            self.assertTrue(inferred["method"] == "unknown" or inferred["f0"] is None)

    def test_unknown_filter_returns_analysis_only_metadata(self):
        inferred = infer_iir_params([1.0], [1.0], FS)

        self.assertEqual(inferred["ftype"], "unknown")
        self.assertEqual(inferred["order"], 0)
        self.assertFalse(inferred["designable"])

    def assert_finite_coefficients(self, b, a):
        self.assertTrue(np.all(np.isfinite(b)))
        self.assertTrue(np.all(np.isfinite(a)))


if __name__ == "__main__":
    unittest.main()
