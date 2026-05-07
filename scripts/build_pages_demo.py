import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy.signal import freqz, tf2sos, tf2zpk


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from iir_filter import design_iir

RESPONSE_POINTS = 1024
DEFAULT_Q = math.sqrt(0.5)
DEFAULT_DESIGN_PARAMS = {
    "ftype": "bandpass",
    "method": "biquad",
    "fs": 48000,
    "f0": 1000,
    "Q": DEFAULT_Q,
    "order": 2,
    "rp": None,
    "rs": None,
}


def build_site(output_dir):
    output_path = Path(output_dir)
    _prepare_output_dir(output_path)
    static_path = output_path / "static"
    static_path.mkdir(parents=True)

    shutil.copy2(REPO_ROOT / "static" / "styles.css", static_path / "styles.css")
    shutil.copy2(REPO_ROOT / "static" / "demo.js", static_path / "demo.js")

    (output_path / "index.html").write_text(_render_html(), encoding="utf-8")
    return output_path


def _prepare_output_dir(output_path):
    output_path.mkdir(parents=True, exist_ok=True)
    for child in output_path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _render_html():
    initial_response = json.dumps(
        _build_initial_design_result(),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    initial_response = initial_response.replace("</", "<\\/")
    design_source = _script_safe_text((REPO_ROOT / "iir_filter" / "design.py").read_text(encoding="utf-8"))
    infer_source = _script_safe_text((REPO_ROOT / "iir_filter" / "infer.py").read_text(encoding="utf-8"))
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>IIR Filter Static Demo</title>
    <script>
      (() => {{
        try {{
          const storedTheme = window.localStorage.getItem("iir-filter-tool:theme");
          const prefersNight = window.matchMedia?.("(prefers-color-scheme: dark)").matches;
          const theme = storedTheme === "day" || storedTheme === "night" ? storedTheme : prefersNight ? "night" : "day";
          document.documentElement.dataset.theme = theme;
        }} catch {{
          document.documentElement.dataset.theme = "day";
        }}
      }})();
    </script>
    <link rel="stylesheet" href="static/styles.css">
  </head>
  <body>
    <header class="app-header">
      <div>
        <p class="eyebrow">IIR Filter Tool</p>
        <h1>IIR Filter Static Demo</h1>
      </div>
      <div class="header-actions">
        <fieldset class="mode-toggle theme-toggle" aria-label="UI theme">
          <legend>UI theme</legend>
          <label>
            <input type="radio" name="theme-mode" value="day" checked>
            <span>Day</span>
          </label>
          <label>
            <input type="radio" name="theme-mode" value="night">
            <span>Night</span>
          </label>
        </fieldset>
        <div id="status" class="status" role="status" aria-live="polite">Static Demo</div>
      </div>
    </header>

    <main class="shell">
      <section class="panel controls-panel" aria-labelledby="design-heading">
        <div class="section-title">
          <h2 id="design-heading">Design Parameters</h2>
          <div class="button-group" aria-label="Design parameter actions">
            <button id="paste-design-json" class="ghost-button" type="button">Paste JSON</button>
            <button id="reset-design" class="ghost-button" type="button">Reset</button>
          </div>
        </div>

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
            <input name="fs" type="number" min="1" step="1" value="48000">
          </label>
          <label>
            <span>Frequency f0 (Hz)</span>
            <input name="f0" type="number" min="1" step="1" value="1000">
          </label>
          <label>
            <span>Q</span>
            <input name="Q" type="number" min="0.0001" step="any" value="{DEFAULT_DESIGN_PARAMS['Q']}">
          </label>
          <label>
            <span>Order</span>
            <input name="order" type="number" min="1" step="1" value="2">
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
          <h2 id="response-heading">Design Response</h2>
          <label class="points-control">
            <span>Points</span>
            <input id="design-response-points" name="design_response_points" type="number" min="2" max="65536" step="1" value="1024">
          </label>
        </div>
        <div class="chart-wrap">
          <canvas id="response-chart" width="960" height="420" aria-label="Design response chart"></canvas>
        </div>
      </section>

      <section class="panel coefficients-panel" aria-labelledby="coefficients-heading">
        <div class="section-title">
          <div class="title-with-mode">
            <h2 id="coefficients-heading">Design Coefficients</h2>
            <fieldset class="mode-toggle" aria-label="Design coefficient mode">
              <legend>Design coefficient mode</legend>
              <label>
                <input type="radio" name="design-coefficient-mode" value="tf" checked>
                <span>tf</span>
              </label>
              <label>
                <input type="radio" name="design-coefficient-mode" value="sos">
                <span>sos</span>
              </label>
              <label>
                <input type="radio" name="design-coefficient-mode" value="zpk">
                <span>zpk</span>
              </label>
            </fieldset>
          </div>
          <div class="button-group" aria-label="Coefficient copy actions">
            <button id="copy-text" class="ghost-button" type="button">Copy Text</button>
            <button id="copy-json" class="ghost-button" type="button">Copy JSON</button>
          </div>
        </div>
        <div id="design-coeff-view" class="coeff-columns">
          <div>
            <h3>b</h3>
            <ul id="b-list" class="coeff-list"></ul>
          </div>
          <div>
            <h3>a</h3>
            <ul id="a-list" class="coeff-list"></ul>
          </div>
        </div>
        <pre id="coeff-json" class="json-block">{{}}</pre>
      </section>

      <section class="panel infer-panel" aria-labelledby="infer-heading">
        <div class="section-title">
          <div class="title-with-mode">
            <h2 id="infer-heading">Inference Coefficients</h2>
            <fieldset class="mode-toggle" aria-label="Inference coefficient mode">
              <legend>Inference coefficient mode</legend>
              <label>
                <input type="radio" name="inference-coefficient-mode" value="tf" checked>
                <span>tf</span>
              </label>
              <label>
                <input type="radio" name="inference-coefficient-mode" value="sos">
                <span>sos</span>
              </label>
              <label>
                <input type="radio" name="inference-coefficient-mode" value="zpk">
                <span>zpk</span>
              </label>
            </fieldset>
          </div>
          <div class="button-group" aria-label="Inference coefficient actions">
            <button id="reset-inference" class="ghost-button" type="button">Reset</button>
          </div>
        </div>
        <form id="infer-form" class="infer-grid">
          <label>
            <span id="inference-coeff-label">Coefficient text (b0,b1,b2,a0,a1,a2)</span>
            <textarea name="coefficients" rows="6" placeholder="b0,b1,b2,a0,a1,a2">[0.08449720532662121, 0.0, -0.08449720532662121, 1.0, -1.815341082704568, 0.8310055893467576]</textarea>
          </label>
          <label>
            <span>Sample Rate (Hz)</span>
            <input name="fs" type="number" min="1" step="1" value="48000">
          </label>
        </form>
      </section>

      <section class="panel inferred-panel" aria-labelledby="inferred-heading">
        <div class="section-title">
          <h2 id="inferred-heading">Inference Parameters</h2>
          <button id="copy-inferred-json" class="ghost-button" type="button">Copy JSON</button>
        </div>
        <dl id="inferred-summary" class="summary-list"></dl>
        <pre id="inferred-json" class="json-block">{{}}</pre>
      </section>

      <section class="panel inference-response-panel" aria-labelledby="inference-response-heading">
        <div class="section-title">
          <h2 id="inference-response-heading">Inference Response</h2>
          <label class="points-control">
            <span>Points</span>
            <input id="inference-response-points" name="inference_response_points" type="number" min="2" max="65536" step="1" value="1024">
          </label>
        </div>
        <div class="chart-wrap">
          <canvas id="inference-response-chart" width="960" height="420" aria-label="Inference response chart"></canvas>
        </div>
      </section>

      <section class="panel about-panel" aria-labelledby="about-heading">
        <div class="about-copy">
          <p class="eyebrow">About Me</p>
          <h2 id="about-heading">Chang Wei Lin</h2>
          <blockquote class="about-quote">
            <p>我愛星空至深，無懼黑夜。</p>
            <p>We have loved the stars too fondly to fear the dark.</p>
            <cite>&mdash; &lt;The Old Astronomer&gt; Sarah Williams</cite>
          </blockquote>
        </div>
        <nav class="social-links" aria-label="About Me links">
          <a class="social-link" href="https://github.com/changweilin" target="_blank" rel="noreferrer" aria-label="GitHub profile" title="GitHub">
            <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
              <path d="M12 .5a12 12 0 0 0-3.79 23.39c.6.11.82-.26.82-.58v-2.03c-3.34.73-4.04-1.61-4.04-1.61-.55-1.39-1.34-1.76-1.34-1.76-1.09-.75.08-.73.08-.73 1.21.08 1.85 1.24 1.85 1.24 1.07 1.84 2.81 1.31 3.5 1 .11-.78.42-1.31.76-1.61-2.67-.3-5.47-1.33-5.47-5.92 0-1.31.47-2.38 1.24-3.22-.12-.3-.54-1.52.12-3.18 0 0 1.01-.32 3.3 1.23a11.5 11.5 0 0 1 6 0c2.29-1.55 3.3-1.23 3.3-1.23.66 1.66.24 2.88.12 3.18.77.84 1.24 1.91 1.24 3.22 0 4.6-2.81 5.62-5.48 5.92.43.37.81 1.1.81 2.22v3.29c0 .32.22.69.82.58A12 12 0 0 0 12 .5Z"></path>
            </svg>
            <span class="sr-only">GitHub</span>
          </a>
          <a class="social-link" href="https://www.linkedin.com/in/wei-lin-chang-ba38049a/" target="_blank" rel="noreferrer" aria-label="LinkedIn profile" title="LinkedIn">
            <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
              <path d="M4.98 3.5a2.5 2.5 0 1 1 0 5 2.5 2.5 0 0 1 0-5ZM3 9h4v12H3V9Zm6.5 0h3.8v1.64h.05c.53-1 1.84-2.05 3.78-2.05 4.04 0 4.79 2.66 4.79 6.12V21h-4v-5.57c0-1.33-.02-3.04-1.85-3.04-1.86 0-2.14 1.45-2.14 2.95V21h-4V9Z"></path>
            </svg>
            <span class="sr-only">LinkedIn</span>
          </a>
          <a class="social-link" href="https://changweilin.github.io/demo_link/" target="_blank" rel="noreferrer" aria-label="Demo links" title="Demo links">
            <img src="https://changweilin.github.io/demo_link/favicon-32.png" alt="" width="22" height="22" loading="lazy" decoding="async">
            <span class="sr-only">Demo links</span>
          </a>
        </nav>
      </section>
    </main>

    <script id="initial-response" type="application/json">{initial_response}</script>
    <script id="design-source" type="text/plain">{design_source}</script>
    <script id="infer-source" type="text/plain">{infer_source}</script>
    <script src="https://cdn.jsdelivr.net/pyodide/v0.29.3/full/pyodide.js"></script>
    <script src="static/demo.js"></script>
  </body>
</html>
"""


def _build_initial_design_result():
    params = DEFAULT_DESIGN_PARAMS.copy()
    b, a = design_iir(params, fs=params["fs"])
    return {
        "params": _json_safe(params),
        "b": _json_safe(b),
        "a": _json_safe(a),
        "coefficients": _json_safe(_coefficient_representations(b, a)),
        "response": _frequency_response(b, a, params["fs"], RESPONSE_POINTS),
    }


def _frequency_response(b, a, fs, points):
    frequency, response = freqz(b, a, worN=points, fs=fs)
    magnitude = np.maximum(np.abs(response), np.finfo(float).tiny)
    magnitude_db = 20 * np.log10(magnitude)
    return {
        "frequency_hz": _json_safe(frequency),
        "magnitude_db": _json_safe(magnitude_db),
    }


def _coefficient_representations(b, a):
    z, p, k = tf2zpk(b, a)
    return {
        "tf": {"b": b, "a": a},
        "sos": tf2sos(b, a),
        "zpk": {"z": z, "p": p, "k": k},
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
