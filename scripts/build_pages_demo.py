import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy.signal import freqz


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from iir_filter import design_iir, infer_iir_params


RESPONSE_POINTS = 1024

DEMO_CASES = [
    {
        "id": "biquad-bandpass",
        "title": "Biquad Bandpass",
        "description": "RBJ biquad bandpass centered at 1 kHz.",
        "params": {
            "ftype": "bandpass",
            "method": "biquad",
            "fs": 48000,
            "f0": 1000,
            "Q": 5,
            "order": 2,
            "rp": None,
            "rs": None,
        },
    },
    {
        "id": "butter-lowpass",
        "title": "Butterworth Lowpass",
        "description": "Fourth-order lowpass with a 2 kHz cutoff.",
        "params": {
            "ftype": "lowpass",
            "method": "butterworth",
            "fs": 48000,
            "f0": 2000,
            "Q": None,
            "order": 4,
            "rp": None,
            "rs": None,
        },
    },
    {
        "id": "elliptic-notch",
        "title": "Elliptic Notch",
        "description": "Narrow 60 Hz rejection example using an elliptic design.",
        "params": {
            "ftype": "notch",
            "method": "elliptic",
            "fs": 48000,
            "f0": 60,
            "Q": 12,
            "order": 2,
            "rp": 1,
            "rs": 45,
        },
    },
]


def build_site(output_dir):
    output_path = Path(output_dir)
    _prepare_output_dir(output_path)
    static_path = output_path / "static"
    static_path.mkdir(parents=True)

    shutil.copy2(REPO_ROOT / "static" / "styles.css", static_path / "styles.css")
    shutil.copy2(REPO_ROOT / "static" / "demo.js", static_path / "demo.js")

    cases = [_build_case(case) for case in DEMO_CASES]
    (output_path / "index.html").write_text(_render_html(cases), encoding="utf-8")
    return output_path


def _prepare_output_dir(output_path):
    output_path.mkdir(parents=True, exist_ok=True)
    for child in output_path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _build_case(case):
    params = case["params"].copy()
    b, a = design_iir(params, fs=params["fs"])
    inferred = infer_iir_params(b, a, params["fs"])

    return {
        "id": case["id"],
        "title": case["title"],
        "description": case["description"],
        "params": _json_safe(params),
        "b": _json_safe(b),
        "a": _json_safe(a),
        "response": _frequency_response(b, a, params["fs"]),
        "inferred": _json_safe(inferred),
    }


def _frequency_response(b, a, fs):
    f, h = freqz(b, a, worN=RESPONSE_POINTS, fs=fs)
    magnitude = np.maximum(np.abs(h), np.finfo(float).tiny)
    magnitude_db = 20 * np.log10(magnitude)
    return {
        "frequency_hz": _json_safe(f),
        "magnitude_db": _json_safe(magnitude_db),
    }


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, complex):
        return {"real": _json_safe(value.real), "imag": _json_safe(value.imag)}
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _render_html(cases):
    data = json.dumps(cases, ensure_ascii=False, separators=(",", ":"))
    data = data.replace("</", "<\\/")
    design_source = _script_safe_text((REPO_ROOT / "iir_filter" / "design.py").read_text(encoding="utf-8"))
    infer_source = _script_safe_text((REPO_ROOT / "iir_filter" / "infer.py").read_text(encoding="utf-8"))
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>IIR Filter Static Demo</title>
    <link rel="stylesheet" href="static/styles.css">
  </head>
  <body>
    <header class="app-header">
      <div>
        <p class="eyebrow">IIR Filter Tool</p>
        <h1>IIR Filter Static Demo</h1>
      </div>
      <div id="status" class="status" role="status" aria-live="polite">Static Demo</div>
    </header>

    <main class="shell">
      <section class="panel controls-panel" aria-labelledby="design-heading">
        <div class="section-title">
          <h2 id="design-heading">Design Parameters</h2>
          <span id="auto-update-indicator" class="meta">Auto updates</span>
        </div>
        <div id="preset-list" class="preset-list"></div>

        <form id="design-form" class="form-grid">
          <label>
            <span>Filter Type</span>
            <select name="ftype">
              <option value="bandpass">bandpass</option>
              <option value="lowpass">lowpass</option>
              <option value="highpass">highpass</option>
              <option value="notch">notch</option>
            </select>
          </label>
          <label>
            <span>Method</span>
            <select name="method">
              <option value="biquad">biquad</option>
              <option value="butterworth">butterworth</option>
              <option value="cheby1">cheby1</option>
              <option value="cheby2">cheby2</option>
              <option value="elliptic">elliptic</option>
              <option value="bessel">bessel</option>
            </select>
          </label>
          <label>
            <span>Sample Rate (Hz)</span>
            <input name="fs" type="number" min="1" step="1">
          </label>
          <label>
            <span>Frequency f0 (Hz)</span>
            <input name="f0" type="number" min="1" step="1">
          </label>
          <label>
            <span>Q</span>
            <input name="Q" type="number" min="0.0001" step="0.1">
          </label>
          <label>
            <span>Order</span>
            <input name="order" type="number" min="1" step="1">
          </label>
          <label>
            <span>Passband Ripple rp (dB)</span>
            <input name="rp" type="number" min="0.0001" step="0.1" placeholder="preset only">
          </label>
          <label>
            <span>Stopband Attenuation rs (dB)</span>
            <input name="rs" type="number" min="0.0001" step="1" placeholder="preset only">
          </label>
        </form>
      </section>

      <section class="panel chart-panel" aria-labelledby="response-heading">
        <div class="section-title">
          <h2 id="response-heading">Magnitude Response</h2>
          <span id="chart-meta" class="meta">0 points</span>
        </div>
        <div class="chart-wrap">
          <canvas id="response-chart" width="960" height="420" aria-label="Magnitude response chart"></canvas>
        </div>
      </section>

      <section class="panel coefficients-panel" aria-labelledby="coefficients-heading">
        <div class="section-title">
          <h2 id="coefficients-heading">Coefficients</h2>
          <button id="copy-json" class="ghost-button" type="button">Copy JSON</button>
        </div>
        <div class="coeff-columns">
          <div>
            <h3>b</h3>
            <ol id="b-list" class="coeff-list"></ol>
          </div>
          <div>
            <h3>a</h3>
            <ol id="a-list" class="coeff-list"></ol>
          </div>
        </div>
        <pre id="coeff-json" class="json-block">{{}}</pre>
      </section>

      <section class="panel infer-panel" aria-labelledby="infer-heading">
        <div class="section-title">
          <h2 id="infer-heading">Coefficient Inference</h2>
          <button id="infer-submit" class="primary-button" type="button">Show Inferred</button>
        </div>
        <form id="infer-form" class="infer-grid">
          <label>
            <span>b coefficients</span>
            <textarea name="b" rows="4" readonly></textarea>
          </label>
          <label>
            <span>a coefficients</span>
            <textarea name="a" rows="4" readonly></textarea>
          </label>
          <label>
            <span>Sample Rate (Hz)</span>
            <input name="fs" readonly>
          </label>
        </form>
      </section>

      <section class="panel inferred-panel" aria-labelledby="inferred-heading">
        <div class="section-title">
          <h2 id="inferred-heading">Inferred Result</h2>
          <button id="apply-inferred" class="ghost-button" type="button">Apply</button>
        </div>
        <dl id="inferred-summary" class="summary-list"></dl>
        <pre id="inferred-json" class="json-block">{{}}</pre>
      </section>
    </main>

    <script id="demo-data" type="application/json">{data}</script>
    <script id="design-source" type="text/plain">{design_source}</script>
    <script id="infer-source" type="text/plain">{infer_source}</script>
    <script src="https://cdn.jsdelivr.net/pyodide/v0.29.3/full/pyodide.js"></script>
    <script src="static/demo.js"></script>
  </body>
</html>
"""


def _script_safe_text(text):
    return text.replace("</", "<\\/")


def main():
    parser = argparse.ArgumentParser(description="Build the static GitHub Pages demo.")
    parser.add_argument("--output", default=REPO_ROOT / "site", type=Path)
    args = parser.parse_args()
    build_site(args.output)
    print(f"Built static demo at {args.output}")


if __name__ == "__main__":
    main()
