import unittest

import numpy as np

from web_app import create_app, _json_safe


class WebAppTests(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = self.app.test_client()

    def test_design_biquad_bandpass_returns_coefficients_and_response(self):
        response = self.client.post(
            "/api/design",
            json={
                "ftype": "bandpass",
                "method": "biquad",
                "fs": 48000,
                "f0": 1000,
                "Q": 5,
                "order": 2,
            },
        )

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(len(data["b"]), 3)
        self.assertEqual(len(data["a"]), 3)
        self.assertTrue(all(np.isfinite(data["b"])))
        self.assertTrue(all(np.isfinite(data["a"])))
        self.assertEqual(len(data["response"]["frequency_hz"]), 1024)
        self.assertEqual(len(data["response"]["magnitude_db"]), 1024)
        self.assertNotIn("inferred", data)

    def test_design_rejects_frequency_at_or_above_nyquist(self):
        response = self.client.post(
            "/api/design",
            json={
                "ftype": "lowpass",
                "method": "butterworth",
                "fs": 48000,
                "f0": 24000,
                "order": 2,
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.get_json())

    def test_infer_known_coefficients_returns_metadata(self):
        design_response = self.client.post(
            "/api/design",
            json={
                "ftype": "bandpass",
                "method": "biquad",
                "fs": 48000,
                "f0": 1000,
                "Q": 5,
                "order": 2,
            },
        )
        coefficients = design_response.get_json()

        response = self.client.post(
            "/api/infer",
            json={
                "b": coefficients["b"],
                "a": coefficients["a"],
                "fs": 48000,
            },
        )

        self.assertEqual(response.status_code, 200)
        inferred = response.get_json()["inferred"]
        self.assertEqual(inferred["ftype"], "bandpass")
        self.assertAlmostEqual(inferred["f0"], 1000, delta=10)
        self.assertEqual(inferred["order"], 2)
        self.assertIn("designable", inferred)
        self.assertEqual(len(response.get_json()["response"]["frequency_hz"]), 1024)
        self.assertEqual(len(response.get_json()["response"]["magnitude_db"]), 1024)

    def test_json_safe_serializes_numpy_and_complex_values(self):
        safe = _json_safe(
            {
                "array": np.array([1.0, 2.0]),
                "scalar": np.float64(3.5),
                "complex": np.complex128(1 + 2j),
                "nan": np.float64(np.nan),
            }
        )

        self.assertEqual(safe["array"], [1.0, 2.0])
        self.assertEqual(safe["scalar"], 3.5)
        self.assertEqual(safe["complex"], {"real": 1.0, "imag": 2.0})
        self.assertIsNone(safe["nan"])


if __name__ == "__main__":
    unittest.main()
