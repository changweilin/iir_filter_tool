const DEFAULT_DESIGN_PARAMS = {
  ftype: "bandpass",
  method: "biquad",
  fs: 48000,
  f0: 1000,
  Q: 5,
  order: 2,
  rp: null,
  rs: null,
};

function parseDemoCases() {
  try {
    const data = JSON.parse(document.querySelector("#demo-data")?.textContent || "[]");
    return Array.isArray(data) ? data : [];
  } catch {
    return [];
  }
}

const demoCases = parseDemoCases();

const demoState = {
  index: 0,
  designCoefficients: null,
  designResponse: null,
  inferenceInferred: null,
  inferenceResponse: null,
  pyodide: null,
  ready: false,
};

const statusEl = document.querySelector("#status");
const designChart = document.querySelector("#response-chart");
const designChartMeta = document.querySelector("#chart-meta");
const inferenceChart = document.querySelector("#inference-response-chart");
const inferenceChartMeta = document.querySelector("#inference-chart-meta");
const designForm = document.querySelector("#design-form");
const inferForm = document.querySelector("#infer-form");
const presetList = document.querySelector("#preset-list");
const inferButton = document.querySelector("#infer-submit");
const autoUpdateIndicator = document.querySelector("#auto-update-indicator");
let autoDesignTimer = null;
let autoDesignPending = false;

function setStatus(message, mode = "ready") {
  statusEl.textContent = message;
  statusEl.classList.toggle("is-error", mode === "error");
  statusEl.classList.toggle("is-working", mode === "working");
}

function setBusy(isBusy) {
  inferButton.disabled = isBusy || !demoState.ready;
  autoUpdateIndicator.textContent = demoState.ready ? (isBusy ? "Updating" : "Auto updates") : "Loading engine";
}

function formatNumber(value) {
  if (value == null) {
    return "";
  }
  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : value.toPrecision(8);
  }
  return String(value);
}

function numberOrNull(value) {
  if (value === "" || value == null) {
    return null;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function positiveNumber(value, name) {
  const parsed = numberOrNull(value);
  if (parsed == null || parsed <= 0) {
    throw new Error(`${name} must be a positive finite value`);
  }
  return parsed;
}

function positiveInteger(value, name) {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(`${name} must be a positive integer`);
  }
  return parsed;
}

function renderList(selector, values) {
  const list = document.querySelector(selector);
  list.innerHTML = "";
  (values || []).forEach((value) => {
    const item = document.createElement("li");
    item.textContent = formatNumber(value);
    list.appendChild(item);
  });
}

function renderSummary(inferred) {
  const summary = document.querySelector("#inferred-summary");
  const fields = ["ftype", "method", "f0", "Q", "order", "fs", "rp", "rs", "gd_dev", "designable"];
  summary.innerHTML = "";
  fields.forEach((field) => {
    const group = document.createElement("div");
    const term = document.createElement("dt");
    const detail = document.createElement("dd");
    term.textContent = field;
    detail.textContent = formatNumber(inferred?.[field]);
    group.append(term, detail);
    summary.appendChild(group);
  });
}

function renderPresetButtons() {
  presetList.innerHTML = "";
  demoCases.forEach((demoCase, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "preset-button";
    button.innerHTML = `<strong>${demoCase.title}</strong><span>${demoCase.description}</span>`;
    button.addEventListener("click", () => renderCase(index));
    presetList.appendChild(button);
  });
}

function setFormValues(params) {
  ["ftype", "method", "fs", "f0", "Q", "order", "rp", "rs"].forEach((field) => {
    if (designForm.elements[field]) {
      designForm.elements[field].value = formatNumber(params[field]);
    }
  });
}

function currentDesignParams() {
  const data = new FormData(designForm);
  return {
    ftype: data.get("ftype"),
    method: data.get("method"),
    fs: positiveNumber(data.get("fs"), "fs"),
    f0: positiveNumber(data.get("f0"), "f0"),
    Q: numberOrNull(data.get("Q")),
    order: positiveInteger(data.get("order"), "order"),
    rp: numberOrNull(data.get("rp")),
    rs: numberOrNull(data.get("rs")),
  };
}

function currentInferPayload() {
  return {
    b: inferForm.elements.b.value,
    a: inferForm.elements.a.value,
    fs: positiveNumber(inferForm.elements.fs.value, "fs"),
  };
}

function applyMethodDefaults() {
  const method = designForm.elements.method.value;
  if (method === "cheby1" && !designForm.elements.rp.value) {
    designForm.elements.rp.value = "1";
  }
  if (method === "cheby2" && !designForm.elements.rs.value) {
    designForm.elements.rs.value = "40";
  }
  if (method === "elliptic") {
    if (!designForm.elements.rp.value) {
      designForm.elements.rp.value = "1";
    }
    if (!designForm.elements.rs.value) {
      designForm.elements.rs.value = "40";
    }
  }
  if (method === "biquad") {
    designForm.elements.order.value = "2";
    designForm.elements.rp.value = "";
    designForm.elements.rs.value = "";
  }
}

function scheduleAutoDesign() {
  if (!demoState.ready) {
    autoDesignPending = true;
    return;
  }
  autoDesignPending = false;
  clearTimeout(autoDesignTimer);
  autoDesignTimer = setTimeout(runDesign, 450);
}

function renderCase(index) {
  if (!demoCases.length) {
    setFormValues(DEFAULT_DESIGN_PARAMS);
    applyMethodDefaults();
    return;
  }

  demoState.index = index;
  const demoCase = demoCases[index];
  setFormValues(demoCase.params);
  applyMethodDefaults();
  renderDesignResult({
    b: demoCase.b,
    a: demoCase.a,
    response: demoCase.response,
  });
  setStatus(demoState.ready ? demoCase.title : "Loading Python", demoState.ready ? "ready" : "working");

  [...presetList.children].forEach((button, buttonIndex) => {
    button.classList.toggle("is-active", buttonIndex === index);
  });
}

function renderDesignResult(data) {
  if (data.b && data.a) {
    demoState.designCoefficients = { b: data.b, a: data.a };
    renderList("#b-list", data.b);
    renderList("#a-list", data.a);
    document.querySelector("#coeff-json").textContent = JSON.stringify(demoState.designCoefficients, null, 2);
  }

  if (data.response) {
    demoState.designResponse = data.response;
    drawChart(designChart, designChartMeta, data.response);
  }

  setBusy(false);
}

function renderInferenceResult(data) {
  if (data.inferred) {
    demoState.inferenceInferred = data.inferred;
    renderSummary(data.inferred);
    document.querySelector("#inferred-json").textContent = JSON.stringify(data.inferred, null, 2);
  }

  if (data.response) {
    demoState.inferenceResponse = data.response;
    drawChart(inferenceChart, inferenceChartMeta, data.response);
  }

  setBusy(false);
}

async function initPyodideRuntime() {
  setBusy(true);
  setStatus("Loading Python", "working");

  if (!globalThis.loadPyodide) {
    throw new Error("Pyodide failed to load. Check the network connection for the CDN assets.");
  }

  const pyodide = await loadPyodide({
    indexURL: "https://cdn.jsdelivr.net/pyodide/v0.29.3/full/",
  });
  setStatus("Loading SciPy", "working");
  await pyodide.loadPackage(["numpy", "scipy"]);

  const designSource = document.querySelector("#design-source").textContent;
  const inferSource = document.querySelector("#infer-source").textContent;
  pyodide.globals.set("DESIGN_SOURCE", designSource);
  pyodide.globals.set("INFER_SOURCE", inferSource);
  pyodide.runPython(`
import json
import sys
import types

design_mod = types.ModuleType("iir_filter_design_static")
exec(DESIGN_SOURCE, design_mod.__dict__)
infer_mod = types.ModuleType("iir_filter_infer_static")
exec(INFER_SOURCE, infer_mod.__dict__)

design_iir = design_mod.design_iir
infer_iir_params = infer_mod.infer_iir_params

def _json_safe(value):
    import numpy as np
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

def _frequency_response(b, a, fs):
    import numpy as np
    from scipy.signal import freqz
    f, h = freqz(b, a, worN=1024, fs=fs)
    magnitude = np.maximum(np.abs(h), np.finfo(float).tiny)
    magnitude_db = 20 * np.log10(magnitude)
    return {
        "frequency_hz": _json_safe(f),
        "magnitude_db": _json_safe(magnitude_db),
    }

def _parse_coefficients(value, name):
    import numpy as np
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(f"Missing required field: {name}")
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            import re
            value = [part for part in re.split(r"[\\s,]+", text) if part]
    coefficients = np.asarray(value, dtype=float)
    if coefficients.ndim != 1 or coefficients.size == 0:
        raise ValueError(f"{name} must be a one-dimensional coefficient array")
    if not np.all(np.isfinite(coefficients)):
        raise ValueError(f"{name} coefficients must be finite")
    return coefficients

def py_design(payload_json):
    payload = json.loads(payload_json)
    fs = float(payload.get("fs", 48000))
    b, a = design_iir(payload, fs=fs)
    return json.dumps(_json_safe({
        "b": b,
        "a": a,
        "response": _frequency_response(b, a, fs),
    }))

def py_infer(payload_json):
    payload = json.loads(payload_json)
    fs = float(payload.get("fs", 48000))
    b = _parse_coefficients(payload.get("b"), "b")
    a = _parse_coefficients(payload.get("a"), "a")
    inferred = infer_iir_params(b, a, fs)
    return json.dumps(_json_safe({
        "inferred": inferred,
        "response": _frequency_response(b, a, fs),
    }))
`);

  demoState.pyodide = pyodide;
  demoState.ready = true;
  setStatus("Ready");
  setBusy(false);
  if (autoDesignPending || !demoState.designResponse) {
    scheduleAutoDesign();
  }
}

async function runDesign() {
  try {
    setBusy(true);
    setStatus("Designing", "working");
    const result = demoState.pyodide.globals.get("py_design")(JSON.stringify(currentDesignParams()));
    renderDesignResult(JSON.parse(result));
    setStatus("Designed");
    [...presetList.children].forEach((button) => button.classList.remove("is-active"));
  } catch (error) {
    setStatus(error.message, "error");
    setBusy(false);
  }
}

async function runInfer() {
  try {
    setBusy(true);
    setStatus("Inferring", "working");
    const result = demoState.pyodide.globals.get("py_infer")(JSON.stringify(currentInferPayload()));
    renderInferenceResult(JSON.parse(result));
    setStatus("Inferred");
  } catch (error) {
    setStatus(error.message, "error");
    setBusy(false);
  }
}

function drawChart(canvas, metaEl, response) {
  const frequencies = response?.frequency_hz || [];
  const magnitudes = response?.magnitude_db || [];
  const ctx = canvas.getContext("2d");
  const pixelRatio = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(320, Math.floor(rect.width * pixelRatio));
  canvas.height = Math.max(260, Math.floor(rect.height * pixelRatio));
  ctx.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);

  const width = rect.width;
  const height = rect.height;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);

  if (!frequencies.length || !magnitudes.length) {
    metaEl.textContent = "0 points";
    return;
  }

  const padding = { top: 24, right: 24, bottom: 42, left: 62 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  const minX = frequencies[0];
  const maxX = frequencies[frequencies.length - 1];
  const displayMagnitudes = magnitudes.map((value) => (Number.isFinite(value) ? Math.max(value, -120) : value));
  const finiteMagnitudes = displayMagnitudes.filter(Number.isFinite);
  const rawMinY = Math.min(...finiteMagnitudes);
  const rawMaxY = Math.max(...finiteMagnitudes);
  const minY = Math.floor((rawMinY - 3) / 10) * 10;
  const maxY = Math.ceil((rawMaxY + 3) / 10) * 10;

  const xScale = (value) => padding.left + ((value - minX) / (maxX - minX || 1)) * plotWidth;
  const yScale = (value) => padding.top + (1 - (value - minY) / (maxY - minY || 1)) * plotHeight;

  ctx.strokeStyle = "#dce4ea";
  ctx.lineWidth = 1;
  ctx.fillStyle = "#66727c";
  ctx.font = "12px Segoe UI, sans-serif";

  for (let i = 0; i <= 4; i += 1) {
    const y = padding.top + (plotHeight * i) / 4;
    const value = maxY - ((maxY - minY) * i) / 4;
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(width - padding.right, y);
    ctx.stroke();
    ctx.fillText(`${value.toFixed(0)} dB`, 10, y + 4);
  }

  for (let i = 0; i <= 4; i += 1) {
    const x = padding.left + (plotWidth * i) / 4;
    const value = minX + ((maxX - minX) * i) / 4;
    ctx.beginPath();
    ctx.moveTo(x, padding.top);
    ctx.lineTo(x, height - padding.bottom);
    ctx.stroke();
    ctx.textAlign = i === 0 ? "left" : i === 4 ? "right" : "center";
    ctx.fillText(`${Math.round(value)} Hz`, x, height - 16);
  }
  ctx.textAlign = "left";

  ctx.strokeStyle = "#172026";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  frequencies.forEach((frequency, index) => {
    const magnitude = displayMagnitudes[index];
    if (!Number.isFinite(magnitude)) {
      return;
    }
    const x = xScale(frequency);
    const y = yScale(magnitude);
    if (index === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  });
  ctx.stroke();

  ctx.strokeStyle = "#006d77";
  ctx.lineWidth = 2;
  ctx.strokeRect(padding.left, padding.top, plotWidth, plotHeight);
  metaEl.textContent = `${frequencies.length} points`;
}

inferButton.addEventListener("click", runInfer);

designForm.addEventListener("input", (event) => {
  if (event.target.name === "method") {
    applyMethodDefaults();
  }
  scheduleAutoDesign();
});

designForm.addEventListener("change", (event) => {
  if (event.target.name === "method") {
    applyMethodDefaults();
  }
  scheduleAutoDesign();
});

document.querySelector("#copy-json").addEventListener("click", async () => {
  if (!demoState.designCoefficients) {
    return;
  }
  await navigator.clipboard.writeText(JSON.stringify(demoState.designCoefficients, null, 2));
  setStatus("Copied");
});

window.addEventListener("resize", () => {
  if (demoState.designResponse) {
    drawChart(designChart, designChartMeta, demoState.designResponse);
  }
  if (demoState.inferenceResponse) {
    drawChart(inferenceChart, inferenceChartMeta, demoState.inferenceResponse);
  }
});

renderPresetButtons();
if (demoCases.length) {
  renderCase(0);
} else {
  setFormValues(DEFAULT_DESIGN_PARAMS);
  applyMethodDefaults();
  setStatus("Loading Python", "working");
}
drawChart(inferenceChart, inferenceChartMeta, null);
setBusy(true);
initPyodideRuntime().catch((error) => {
  demoState.ready = false;
  setBusy(false);
  inferButton.disabled = true;
  setStatus(error.message, "error");
});
