import json
import re
import shutil
import unittest
from pathlib import Path

from scripts.build_pages_demo import DEMO_CASES, build_site


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
        self.assertIn('id="auto-update-indicator"', html)
        self.assertIn('id="inference-response-chart"', html)
        self.assertNotIn('id="design-submit"', html)
        self.assertNotIn('id="apply-inferred"', html)
        self.assertNotIn('input name="ftype" readonly', html)
        self.assertIn("pyodide/v0.29.3/full/pyodide.js", html)
        self.assertIn('script id="design-source"', html)
        self.assertIn('script id="infer-source"', html)
        self.assertIn("static/demo.js", html)

    def test_demo_data_contains_response_and_coefficients(self):
        output = build_site(self.output_dir)
        html = (output / "index.html").read_text(encoding="utf-8")
        match = re.search(
            r'<script id="demo-data" type="application/json">(.*?)</script>',
            html,
            re.S,
        )
        self.assertIsNotNone(match)

        data = json.loads(match.group(1))
        self.assertEqual(len(data), len(DEMO_CASES))
        self.assertEqual(len(data[0]["b"]), 3)
        self.assertEqual(len(data[0]["a"]), 3)
        self.assertEqual(len(data[0]["response"]["frequency_hz"]), 1024)
        self.assertEqual(len(data[0]["response"]["magnitude_db"]), 1024)
        self.assertNotIn("inferred", data[0])


if __name__ == "__main__":
    unittest.main()
