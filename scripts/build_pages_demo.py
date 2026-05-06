import argparse
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


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
          <div class="button-group" aria-label="Design parameter actions">
            <button id="paste-design-json" class="ghost-button" type="button">Paste JSON</button>
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
            <input name="Q" type="number" min="0.0001" step="0.1" value="5">
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
          <h2 id="coefficients-heading">Design Coefficients</h2>
          <div class="button-group" aria-label="Coefficient copy actions">
            <button id="copy-text" class="ghost-button" type="button">Copy Text</button>
            <button id="copy-json" class="ghost-button" type="button">Copy JSON</button>
          </div>
        </div>
        <div class="coeff-columns">
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
          <h2 id="infer-heading">Inference Coefficient</h2>
        </div>
        <form id="infer-form" class="infer-grid">
          <label>
            <span>Coefficient text (b0,b1,b2,a0,a1,a2)</span>
            <textarea name="coefficients" rows="6" placeholder="b0,b1,b2,a0,a1,a2">[0.01276221, 0, -0.01276221, 1, -1.95676142, 0.97447558]</textarea>
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
    </main>

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
