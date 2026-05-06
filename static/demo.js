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

const demoState = {
  designCoefficients: null,
  designResponse: null,
  inferenceInferred: null,
  inferenceResponse: null,
  pyodide: null,
  ready: false,
};

const statusEl = document.querySelector("#status");
const designChart = document.querySelector("#response-chart");
const designResponsePointsInput = document.querySelector("#design-response-points");
const inferenceChart = document.querySelector("#inference-response-chart");
const inferenceResponsePointsInput = document.querySelector("#inference-response-points");
const designForm = document.querySelector("#design-form");
const inferForm = document.querySelector("#infer-form");
let autoDesignTimer = null;
let autoDesignPending = false;
let autoInferTimer = null;
let autoInferPending = false;
let lastCopiedInferredJson = "";

function parseInitialResponse() {
  try {
    const data = JSON.parse(document.querySelector("#initial-response")?.textContent || "null");
    return data && typeof data === "object" && !Array.isArray(data) ? data : null;
  } catch {
    return null;
  }
}

function setStatus(message, mode = "ready") {
  statusEl.textContent = message;
  statusEl.classList.toggle("is-error", mode === "error");
  statusEl.classList.toggle("is-working", mode === "working");
}

function setBusy(isBusy) {
  document.body.classList.toggle("is-busy", isBusy);
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

function parseNumberSequence(text) {
  const trimmed = String(text || "").trim();
  if (!trimmed) {
    throw new Error("Coefficient text is required");
  }

  try {
    const parsed = JSON.parse(trimmed);
    if (Array.isArray(parsed)) {
      return parsed.map((value) => Number(value));
    }
  } catch {
    // Fall back to comma, semicolon, or whitespace separated values.
  }

  return trimmed
    .replace(/^\[/, "")
    .replace(/\]$/, "")
    .split(/[\s,;]+/)
    .filter(Boolean)
    .map((value) => Number(value));
}

function parseCoefficientText(text) {
  const values = parseNumberSequence(text);
  if (!values.length || values.some((value) => !Number.isFinite(value))) {
    throw new Error("Coefficient text must contain only finite numbers");
  }
  if (values.length % 2 !== 0) {
    throw new Error("Coefficient text must contain the same number of b and a values");
  }

  const splitIndex = values.length / 2;
  return {
    b: values.slice(0, splitIndex),
    a: values.slice(splitIndex),
  };
}

function coefficientText(coefficients) {
  return [...coefficients.b, ...coefficients.a].map(formatNumber).join(",");
}

async function writeClipboardText(text) {
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
      return;
    }
  } catch {
    // Fall back for file:// previews or browsers with restricted clipboard APIs.
  }

  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "");
  textarea.style.position = "fixed";
  textarea.style.inset = "0 auto auto 0";
  textarea.style.opacity = "0";
  document.body.appendChild(textarea);
  textarea.select();
  const copied = document.execCommand("copy");
  textarea.remove();
  if (!copied) {
    throw new Error("Clipboard is unavailable");
  }
}

async function readClipboardText(fallbackText = "") {
  try {
    if (navigator.clipboard?.readText) {
      return await navigator.clipboard.readText();
    }
  } catch {
    // Use the last in-page inferred JSON copy when direct clipboard reads are restricted.
  }

  if (fallbackText) {
    return fallbackText;
  }
  throw new Error("Clipboard is unavailable");
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

function responsePoints(input) {
  const points = positiveInteger(input.value, "response_points");
  if (points < 2 || points > 65536) {
    throw new Error("response_points must be between 2 and 65536");
  }
  return points;
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

function setFormValues(params) {
  ["ftype", "method", "fs", "f0", "Q", "order", "rp", "rs"].forEach((field) => {
    if (designForm.elements[field]) {
      designForm.elements[field].value = formatNumber(params[field]);
    }
  });
}

function setControlValue(control, value, field) {
  const displayValue = value == null ? "" : String(value);
  if (control.tagName === "SELECT") {
    const hasOption = [...control.options].some((option) => option.value === displayValue);
    if (!hasOption) {
      throw new Error(`Unsupported ${field}: ${displayValue}`);
    }
  }
  control.value = displayValue;
}

function applyDesignParameters(params) {
  if (!params || typeof params !== "object" || Array.isArray(params)) {
    throw new Error("JSON must be an object");
  }

  const designFields = ["ftype", "method", "fs", "f0", "Q", "order", "rp", "rs"];
  let appliedCount = 0;
  designFields.forEach((field) => {
    if (Object.hasOwn(params, field) && designForm.elements[field]) {
      setControlValue(designForm.elements[field], params[field], field);
      appliedCount += 1;
    }
  });

  if (!appliedCount) {
    throw new Error("JSON does not contain design parameters");
  }
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
    response_points: responsePoints(designResponsePointsInput),
  };
}

function currentInferPayload() {
  const coefficients = parseCoefficientText(inferForm.elements.coefficients.value);
  return {
    b: coefficients.b,
    a: coefficients.a,
    fs: positiveNumber(inferForm.elements.fs.value, "fs"),
    response_points: responsePoints(inferenceResponsePointsInput),
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

function scheduleAutoInfer() {
  if (!demoState.ready) {
    autoInferPending = true;
    return;
  }
  autoInferPending = false;
  clearTimeout(autoInferTimer);
  autoInferTimer = setTimeout(runInfer, 450);
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
    drawChart(designChart, data.response);
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
    drawChart(inferenceChart, data.response);
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

def _frequency_response(b, a, fs, points=1024):
    import numpy as np
    from scipy.signal import freqz
    f, h = freqz(b, a, worN=points, fs=fs)
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

def _response_points(value):
    points = int(value if value is not None else 1024)
    if points < 2 or points > 65536:
        raise ValueError("response_points must be between 2 and 65536")
    return points

def py_design(payload_json):
    payload = json.loads(payload_json)
    fs = float(payload.get("fs", 48000))
    response_points = _response_points(payload.get("response_points"))
    b, a = design_iir(payload, fs=fs)
    return json.dumps(_json_safe({
        "b": b,
        "a": a,
        "response": _frequency_response(b, a, fs, response_points),
    }))

def py_infer(payload_json):
    payload = json.loads(payload_json)
    fs = float(payload.get("fs", 48000))
    response_points = _response_points(payload.get("response_points"))
    b = _parse_coefficients(payload.get("b"), "b")
    a = _parse_coefficients(payload.get("a"), "a")
    inferred = infer_iir_params(b, a, fs)
    return json.dumps(_json_safe({
        "inferred": inferred,
        "response": _frequency_response(b, a, fs, response_points),
    }))
`);

  demoState.pyodide = pyodide;
  demoState.ready = true;
  setStatus("Ready");
  setBusy(false);
  if (autoDesignPending || !demoState.designResponse) {
    scheduleAutoDesign();
  }
  if (autoInferPending || !demoState.inferenceResponse) {
    scheduleAutoInfer();
  }
}

async function runDesign() {
  try {
    setBusy(true);
    setStatus("Designing", "working");
    const result = demoState.pyodide.globals.get("py_design")(JSON.stringify(currentDesignParams()));
    renderDesignResult(JSON.parse(result));
    setStatus("Designed");
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

async function pasteDesignJson() {
  try {
    const text = await readClipboardText(lastCopiedInferredJson);
    const parsed = JSON.parse(text);
    applyDesignParameters(parsed.inferred || parsed);
    applyMethodDefaults();
    setStatus("Pasted JSON");
    scheduleAutoDesign();
  } catch (error) {
    setStatus(error.message, "error");
  }
}

function drawChart(canvas, response) {
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
}

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
  await writeClipboardText(JSON.stringify(demoState.designCoefficients, null, 2));
  setStatus("Copied");
});

document.querySelector("#copy-text").addEventListener("click", async () => {
  if (!demoState.designCoefficients) {
    return;
  }
  await writeClipboardText(coefficientText(demoState.designCoefficients));
  setStatus("Copied text");
});

document.querySelector("#copy-inferred-json").addEventListener("click", async () => {
  if (!demoState.inferenceInferred) {
    return;
  }
  lastCopiedInferredJson = JSON.stringify(demoState.inferenceInferred, null, 2);
  await writeClipboardText(lastCopiedInferredJson);
  setStatus("Copied inferred JSON");
});

document.querySelector("#paste-design-json").addEventListener("click", pasteDesignJson);

designResponsePointsInput.addEventListener("input", scheduleAutoDesign);
designResponsePointsInput.addEventListener("change", scheduleAutoDesign);
inferForm.addEventListener("input", scheduleAutoInfer);
inferForm.addEventListener("change", scheduleAutoInfer);
inferenceResponsePointsInput.addEventListener("input", scheduleAutoInfer);
inferenceResponsePointsInput.addEventListener("change", scheduleAutoInfer);

window.addEventListener("resize", () => {
  if (demoState.designResponse) {
    drawChart(designChart, demoState.designResponse);
  }
  if (demoState.inferenceResponse) {
    drawChart(inferenceChart, demoState.inferenceResponse);
  }
});

const initialResponse = parseInitialResponse();
setFormValues(initialResponse?.params || DEFAULT_DESIGN_PARAMS);
applyMethodDefaults();
setStatus("Loading Python", "working");
if (initialResponse) {
  renderDesignResult(initialResponse);
}
drawChart(inferenceChart, null);
setBusy(true);
initPyodideRuntime().catch((error) => {
  demoState.ready = false;
  setBusy(false);
  setStatus(error.message, "error");
});
