import unittest

import numpy as np

from web_app import create_app, _json_safe


class WebAppTests(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = self.app.test_client()

    def test_index_contains_about_me_section(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn('name="theme-mode"', html)
        self.assertIn('value="day"', html)
        self.assertIn('value="night"', html)
        self.assertIn('id="reset-design"', html)
        self.assertIn('id="reset-inference"', html)
        self.assertIn('name="Q" type="number" min="0.0001" step="any" value="0.7071067811865476"', html)
        self.assertIn("About Me", html)
        self.assertIn("Chang Wei Lin", html)
        self.assertIn("我愛星空至深，無懼黑夜。", html)
        self.assertIn("We have loved the stars too fondly to fear the dark.", html)
        self.assertIn("https://github.com/changweilin", html)
        self.assertIn("https://www.linkedin.com/in/wei-lin-chang-ba38049a/", html)
        self.assertIn("https://changweilin.github.io/demo_link/", html)
        self.assertIn("https://changweilin.github.io/demo_link/favicon-32.png", html)

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

    def test_design_returns_tf_sos_and_zpk_coefficients(self):
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
        coefficients = response.get_json()["coefficients"]
        self.assertEqual(set(coefficients), {"tf", "sos", "zpk"})
        self.assertEqual(len(coefficients["tf"]["b"]), 3)
        self.assertEqual(len(coefficients["tf"]["a"]), 3)
        self.assertEqual(len(coefficients["sos"][0]), 6)
        self.assertIn("z", coefficients["zpk"])
        self.assertIn("p", coefficients["zpk"])
        self.assertIn("k", coefficients["zpk"])

    def test_design_accepts_custom_response_points(self):
        response = self.client.post(
            "/api/design",
            json={
                "ftype": "bandpass",
                "method": "biquad",
                "fs": 48000,
                "f0": 1000,
                "Q": 5,
                "order": 2,
                "response_points": 256,
            },
        )

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(len(data["response"]["frequency_hz"]), 256)
        self.assertEqual(len(data["response"]["magnitude_db"]), 256)

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
                "response_points": 512,
            },
        )

        self.assertEqual(response.status_code, 200)
        inferred = response.get_json()["inferred"]
        self.assertEqual(inferred["ftype"], "bandpass")
        self.assertAlmostEqual(inferred["f0"], 1000, delta=10)
        self.assertEqual(inferred["order"], 2)
        self.assertIn("designable", inferred)
        self.assertEqual(len(response.get_json()["response"]["frequency_hz"]), 512)
        self.assertEqual(len(response.get_json()["response"]["magnitude_db"]), 512)

    def test_infer_accepts_sos_coefficients(self):
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
        sos = design_response.get_json()["coefficients"]["sos"]

        response = self.client.post(
            "/api/infer",
            json={
                "coefficient_mode": "sos",
                "sos": sos,
                "fs": 48000,
            },
        )

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["inferred"]["ftype"], "bandpass")
        self.assertIn("coefficients", data)

    def test_infer_accepts_zpk_coefficients(self):
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
        zpk = design_response.get_json()["coefficients"]["zpk"]

        response = self.client.post(
            "/api/infer",
            json={
                "coefficient_mode": "zpk",
                "zpk": zpk,
                "fs": 48000,
            },
        )

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["inferred"]["ftype"], "bandpass")
        self.assertAlmostEqual(data["inferred"]["f0"], 1000, delta=10)

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
