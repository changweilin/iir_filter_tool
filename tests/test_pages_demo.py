import shutil
import unittest
from pathlib import Path

from scripts.build_pages_demo import build_site


class PagesDemoTests(unittest.TestCase):
    def setUp(self):
        self.output_dir = Path("site-test-output")

    def tearDown(self):
        shutil.rmtree(self.output_dir, ignore_errors=True)

    def test_build_site_writes_editable_static_demo(self):
        output = build_site(self.output_dir)

        self.assertTrue((output / "index.html").is_file())
        self.assertTrue((output / "static" / "styles.css").is_file())
        self.assertTrue((output / "static" / "demo.js").is_file())

        html = (output / "index.html").read_text(encoding="utf-8")
        self.assertIn("IIR Filter Static Demo", html)
        self.assertIn('select name="ftype"', html)
        self.assertIn('input name="fs" type="number" min="1" step="1" value="48000"', html)
        self.assertIn('input name="f0" type="number" min="1" step="1" value="1000"', html)
        self.assertNotIn("Auto updates", html)
        self.assertNotIn('id="auto-update-indicator"', html)
        self.assertIn("Design Coefficients", html)
        self.assertIn("Design Response", html)
        self.assertIn("Inference Coefficient", html)
        self.assertIn("Inference Parameters", html)
        self.assertIn('id="design-response-points"', html)
        self.assertIn('id="inference-response-points"', html)
        self.assertIn('id="inference-response-chart"', html)
        self.assertIn('id="paste-design-json"', html)
        self.assertIn('id="copy-inferred-json"', html)
        self.assertIn('id="copy-text"', html)
        self.assertIn('id="copy-json"', html)
        self.assertIn('id="b-list" class="coeff-list"', html)
        self.assertIn('id="a-list" class="coeff-list"', html)
        self.assertNotIn('start="0"', html)
        self.assertIn('name="coefficients"', html)
        self.assertNotIn('id="design-submit"', html)
        self.assertNotIn('id="infer-submit"', html)
        self.assertNotIn('id="apply-inferred"', html)
        self.assertNotIn('id="preset-list"', html)
        self.assertNotIn('id="demo-data"', html)
        self.assertNotIn("Biquad Bandpass", html)
        self.assertNotIn("Butterworth Lowpass", html)
        self.assertNotIn("Elliptic Notch", html)
        self.assertNotIn('input name="ftype" readonly', html)
        self.assertIn("pyodide/v0.29.3/full/pyodide.js", html)
        self.assertIn('script id="design-source"', html)
        self.assertIn('script id="infer-source"', html)
        self.assertIn("static/demo.js", html)


if __name__ == "__main__":
    unittest.main()
