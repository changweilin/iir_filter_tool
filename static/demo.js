const DEFAULT_DESIGN_PARAMS = {
  ftype: "bandpass",
  method: "biquad",
  fs: 48000,
  f0: 1000,
  Q: 0.7071067811865476,
  order: 2,
  rp: null,
  rs: null,
};

const FORM_STORAGE_KEY = "iir-filter-tool:form-state:v3";
const THEME_STORAGE_KEY = "iir-filter-tool:theme";
const THEME_MODES = new Set(["day", "night"]);
const COEFFICIENT_MODE_NAMES = {
  design: "design-coefficient-mode",
  inference: "inference-coefficient-mode",
};
const DEFAULT_RESPONSE_POINTS = "1024";
const DEFAULT_INFERENCE_MODE = "tf";
const INFERENCE_MODE_DETAILS = {
  tf: {
    label: "Coefficient text (b0,b1,b2,a0,a1,a2)",
    placeholder: "b0,b1,b2,a0,a1,a2",
    sample: "[0.08449720532662121, 0.0, -0.08449720532662121, 1.0, -1.815341082704568, 0.8310055893467576]",
  },
  sos: {
    label: "Second-order sections (JSON rows)",
    placeholder: "[[b0,b1,b2,a0,a1,a2]]",
    sample: "[[0.08449720532662121, 0.0, -0.08449720532662121, 1.0, -1.8153410827045682, 0.8310055893467578]]",
  },
  zpk: {
    label: "Zeros / poles / gain (JSON)",
    placeholder: '{"z":[-1,1],"p":[{"real":0.9076705413522841,"imag":0.0844972053266221},{"real":0.9076705413522841,"imag":-0.0844972053266221}],"k":0.08449720532662121}',
    sample: '{"z":[-1,1],"p":[{"real":0.9076705413522841,"imag":0.0844972053266221},{"real":0.9076705413522841,"imag":-0.0844972053266221}],"k":0.08449720532662121}',
  },
};

const demoState = {
  designCoefficients: null,
  designResponse: null,
  inferenceInferred: null,
  inferenceCoefficients: null,
  inferenceResponse: null,
  pyodide: null,
  ready: false,
};

const statusEl = document.querySelector("#status");
const themeControls = document.querySelectorAll('input[name="theme-mode"]');
const designChart = document.querySelector("#response-chart");
const designResponsePointsInput = document.querySelector("#design-response-points");
const inferenceChart = document.querySelector("#inference-response-chart");
const inferenceResponsePointsInput = document.querySelector("#inference-response-points");
const designForm = document.querySelector("#design-form");
const inferForm = document.querySelector("#infer-form");
const designCoefficientView = document.querySelector("#design-coeff-view");
const inferenceCoeffLabel = document.querySelector("#inference-coeff-label");
const inferenceCoeffTextarea = inferForm.elements.coefficients;
let autoDesignTimer = null;
let autoDesignPending = false;
let autoInferTimer = null;
let autoInferPending = false;
let lastCopiedInferredJson = "";
let lastInferenceMode = "tf";

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

function preferredThemeMode() {
  const currentTheme = document.documentElement.dataset.theme;
  if (THEME_MODES.has(currentTheme)) {
    return currentTheme;
  }

  try {
    const storedTheme = window.localStorage.getItem(THEME_STORAGE_KEY);
    if (THEME_MODES.has(storedTheme)) {
      return storedTheme;
    }
  } catch {
    // Some embedded or file previews can block localStorage.
  }

  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "night" : "day";
}

function setThemeMode(mode) {
  const theme = THEME_MODES.has(mode) ? mode : "day";
  document.documentElement.dataset.theme = theme;
  themeControls.forEach((control) => {
    control.checked = control.value === theme;
  });
  return theme;
}

function persistThemeMode(mode) {
  try {
    window.localStorage.setItem(THEME_STORAGE_KEY, mode);
  } catch {
    // Some embedded or file previews can block localStorage.
  }
}

function redrawChartsForTheme() {
  if (demoState.designResponse) {
    drawChart(designChart, demoState.designResponse);
  }
  drawChart(inferenceChart, demoState.inferenceResponse);
}

function applyThemeMode(mode) {
  const theme = setThemeMode(mode);
  persistThemeMode(theme);
  redrawChartsForTheme();
}

function setBusy(isBusy) {
  document.body.classList.toggle("is-busy", isBusy);
}

function formatNumber(value) {
  if (value == null) {
    return "";
  }
  if (isComplexObject(value)) {
    return formatComplex(value);
  }
  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : value.toPrecision(8);
  }
  return String(value);
}

function isComplexObject(value) {
  return value && typeof value === "object" && !Array.isArray(value) && ("real" in value || "imag" in value);
}

function formatComplex(value) {
  const real = Number(value?.real || 0);
  const imag = Number(value?.imag || 0);
  if (!Number.isFinite(real) || !Number.isFinite(imag)) {
    return String(value);
  }
  if (Math.abs(imag) < 1e-14) {
    return formatNumber(real);
  }
  if (Math.abs(real) < 1e-14) {
    return `${formatNumber(imag)}j`;
  }
  const sign = imag >= 0 ? "+" : "-";
  return `${formatNumber(real)} ${sign} ${formatNumber(Math.abs(imag))}j`;
}

function formatCoefficientValue(value) {
  if (Array.isArray(value)) {
    return `[${value.map(formatCoefficientValue).join(", ")}]`;
  }
  return formatNumber(value);
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

function parseJsonText(text, name) {
  const trimmed = String(text || "").trim();
  if (!trimmed) {
    throw new Error(`${name} is required`);
  }
  try {
    return JSON.parse(trimmed);
  } catch (error) {
    throw new Error(`${name} must be valid JSON`);
  }
}

function parseNumberArray(value, name) {
  if (!Array.isArray(value)) {
    throw new Error(`${name} must be an array`);
  }
  const values = value.map((item) => Number(item));
  if (!values.length || values.some((item) => !Number.isFinite(item))) {
    throw new Error(`${name} must contain only finite numbers`);
  }
  return values;
}

function parseTfInput(text) {
  const trimmed = String(text || "").trim();
  try {
    const parsed = JSON.parse(trimmed);
    const tf = parsed?.tf || parsed;
    if (tf && typeof tf === "object" && !Array.isArray(tf) && Array.isArray(tf.b) && Array.isArray(tf.a)) {
      return {
        coefficient_mode: "tf",
        b: parseNumberArray(tf.b, "b"),
        a: parseNumberArray(tf.a, "a"),
      };
    }
  } catch {
    // Fall through to the compact b/a sequence format.
  }

  return {
    coefficient_mode: "tf",
    ...parseCoefficientText(trimmed),
  };
}

function parseSosInput(text) {
  const parsed = parseJsonText(text, "SOS coefficients");
  const rawRows = parsed?.sos || parsed;
  if (!Array.isArray(rawRows)) {
    throw new Error("SOS coefficients must be an array");
  }

  const rows = rawRows.every((row) => !Array.isArray(row))
    ? chunkSosRow(parseNumberArray(rawRows, "sos"))
    : rawRows.map((row) => parseNumberArray(row, "sos row"));

  if (!rows.length || rows.some((row) => row.length !== 6)) {
    throw new Error("Each SOS row must contain 6 numbers");
  }
  return {
    coefficient_mode: "sos",
    sos: rows,
  };
}

function chunkSosRow(values) {
  if (values.length % 6 !== 0) {
    throw new Error("SOS coefficient count must be a multiple of 6");
  }
  const rows = [];
  for (let index = 0; index < values.length; index += 6) {
    rows.push(values.slice(index, index + 6));
  }
  return rows;
}

function parseZpkInput(text) {
  const parsed = parseJsonText(text, "ZPK coefficients");
  let zpk = parsed?.zpk || parsed;
  if (Array.isArray(zpk) && zpk.length === 3) {
    zpk = { z: zpk[0], p: zpk[1], k: zpk[2] };
  }
  if (!zpk || typeof zpk !== "object" || Array.isArray(zpk)) {
    throw new Error("ZPK coefficients must be an object or [z, p, k] array");
  }

  return {
    coefficient_mode: "zpk",
    z: parseComplexArray(zpk.z ?? zpk.zeros ?? [], "z"),
    p: parseComplexArray(zpk.p ?? zpk.poles ?? [], "p"),
    k: parseComplexValue(zpk.k ?? zpk.gain ?? 1, "k"),
  };
}

function parseComplexArray(value, name) {
  if (!Array.isArray(value)) {
    throw new Error(`${name} must be an array`);
  }
  return value.map((item) => parseComplexValue(item, name));
}

function parseComplexValue(value, name) {
  if (typeof value === "number") {
    if (!Number.isFinite(value)) {
      throw new Error(`${name} values must be finite`);
    }
    return value;
  }
  if (typeof value === "string") {
    return value;
  }
  if (Array.isArray(value) && value.length === 2) {
    const real = Number(value[0]);
    const imag = Number(value[1]);
    if (!Number.isFinite(real) || !Number.isFinite(imag)) {
      throw new Error(`${name} values must be finite`);
    }
    return { real, imag };
  }
  if (value && typeof value === "object") {
    const real = Number(value.real ?? value.re ?? 0);
    const imag = Number(value.imag ?? value.im ?? 0);
    if (!Number.isFinite(real) || !Number.isFinite(imag)) {
      throw new Error(`${name} values must be finite`);
    }
    return { real, imag };
  }
  throw new Error(`${name} values must be numbers or complex objects`);
}

function parseCoefficientInput(text, mode) {
  if (mode === "sos") {
    return parseSosInput(text);
  }
  if (mode === "zpk") {
    return parseZpkInput(text);
  }
  return parseTfInput(text);
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

function selectedCoefficientMode(name) {
  return document.querySelector(`input[name="${name}"]:checked`)?.value || "tf";
}

function selectedDesignCoefficientMode() {
  return selectedCoefficientMode(COEFFICIENT_MODE_NAMES.design);
}

function selectedInferenceCoefficientMode() {
  return selectedCoefficientMode(COEFFICIENT_MODE_NAMES.inference);
}

function coefficientRepresentations(data) {
  if (data?.coefficients) {
    return data.coefficients;
  }
  if (data?.b && data?.a) {
    return { tf: { b: data.b, a: data.a } };
  }
  return null;
}

function coefficientGroups(coefficients, mode) {
  if (!coefficients) {
    return [];
  }
  if (mode === "sos") {
    return (coefficients.sos || []).map((row, index) => ({
      title: `section ${index + 1}`,
      values: row,
    }));
  }
  if (mode === "zpk") {
    const zpk = coefficients.zpk || {};
    return [
      { title: "z", values: zpk.z || [] },
      { title: "p", values: zpk.p || [] },
      { title: "k", values: [zpk.k] },
    ];
  }
  return [
    { title: "b", id: "b-list", values: coefficients.tf?.b || [] },
    { title: "a", id: "a-list", values: coefficients.tf?.a || [] },
  ];
}

function renderCoefficientView(container, coefficients, mode) {
  container.innerHTML = "";
  coefficientGroups(coefficients, mode).forEach((group) => {
    const column = document.createElement("div");
    const title = document.createElement("h3");
    const list = document.createElement("ul");
    title.textContent = group.title;
    list.className = "coeff-list";
    if (group.id) {
      list.id = group.id;
    }
    (group.values || []).forEach((value) => {
      const item = document.createElement("li");
      item.textContent = formatCoefficientValue(value);
      list.appendChild(item);
    });
    column.append(title, list);
    container.appendChild(column);
  });
}

function selectedRepresentation(coefficients, mode) {
  if (!coefficients) {
    return null;
  }
  if (mode === "sos") {
    return coefficients.sos || null;
  }
  if (mode === "zpk") {
    return coefficients.zpk || null;
  }
  return coefficients.tf || null;
}

function coefficientTextForMode(coefficients, mode) {
  const representation = selectedRepresentation(coefficients, mode);
  if (!representation) {
    return "";
  }
  if (mode === "tf") {
    return coefficientText(representation);
  }
  if (mode === "sos") {
    return (representation || []).map((row) => row.map(formatCoefficientValue).join(",")).join("\n");
  }
  return JSON.stringify(representation, null, 2);
}

function renderDesignCoefficients() {
  const mode = selectedDesignCoefficientMode();
  const representation = selectedRepresentation(demoState.designCoefficients, mode);
  renderCoefficientView(designCoefficientView, demoState.designCoefficients, mode);
  document.querySelector("#coeff-json").textContent = JSON.stringify(representation || {}, null, 2);
}

function coefficientInputText(mode, coefficients) {
  const text = coefficientTextForMode(coefficients, mode);
  if (text) {
    return mode === "sos" ? JSON.stringify(coefficients.sos || [], null, 2) : text;
  }
  return INFERENCE_MODE_DETAILS[mode].sample;
}

function updateInferenceModeUi(useCurrentCoefficients = false) {
  const mode = selectedInferenceCoefficientMode();
  const previousDetails = INFERENCE_MODE_DETAILS[lastInferenceMode];
  const currentText = inferenceCoeffTextarea.value.trim();
  applyInferenceModeDetails(mode);

  if (useCurrentCoefficients && demoState.inferenceCoefficients?.[mode]) {
    inferenceCoeffTextarea.value = coefficientInputText(mode, demoState.inferenceCoefficients);
  } else if (!currentText || currentText === previousDetails.sample) {
    inferenceCoeffTextarea.value = INFERENCE_MODE_DETAILS[mode].sample;
  }
  lastInferenceMode = mode;
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
      designForm.elements[field].value = params[field] == null ? "" : String(params[field]);
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

function setRadioValue(name, value) {
  document.querySelectorAll(`input[name="${name}"]`).forEach((control) => {
    control.checked = control.value === value;
  });
}

function applyInferenceModeDetails(mode) {
  const details = INFERENCE_MODE_DETAILS[mode];
  inferenceCoeffLabel.textContent = details.label;
  inferenceCoeffTextarea.placeholder = details.placeholder;
}

function persistentControls() {
  return [...designForm.elements, designResponsePointsInput, ...inferForm.elements, inferenceResponsePointsInput].filter(
    (control) => control?.name,
  );
}

function controlStorageKey(control) {
  const formId = control.closest("form")?.id;
  return formId ? `${formId}:${control.name}` : control.id || control.name;
}

function savedControlValues() {
  try {
    const stored = window.localStorage.getItem(FORM_STORAGE_KEY);
    const values = stored ? JSON.parse(stored) : null;
    return values && typeof values === "object" && !Array.isArray(values) ? values : null;
  } catch {
    return null;
  }
}

function persistControlValues() {
  try {
    const values = {};
    persistentControls().forEach((control) => {
      values[controlStorageKey(control)] = control.value;
    });
    window.localStorage.setItem(FORM_STORAGE_KEY, JSON.stringify(values));
  } catch {
    // Some embedded or file previews can block localStorage.
  }
}

function restoreControlValues() {
  const values = savedControlValues();
  if (!values) {
    return;
  }

  persistentControls().forEach((control) => {
    const key = controlStorageKey(control);
    if (!Object.hasOwn(values, key)) {
      return;
    }
    try {
      setControlValue(control, values[key], control.name);
    } catch {
      // Ignore stale saved select options from an older app version.
    }
  });
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
  const coefficients = parseCoefficientInput(inferForm.elements.coefficients.value, selectedInferenceCoefficientMode());
  return {
    ...coefficients,
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
  const coefficients = coefficientRepresentations(data);
  if (coefficients) {
    demoState.designCoefficients = coefficients;
    renderDesignCoefficients();
  }

  if (data.response) {
    demoState.designResponse = data.response;
    drawChart(designChart, data.response);
  }

  setBusy(false);
}

function renderInferenceResult(data) {
  const coefficients = coefficientRepresentations(data);
  if (coefficients) {
    demoState.inferenceCoefficients = coefficients;
  }

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

def _coefficient_representations(b, a):
    from scipy.signal import tf2sos, tf2zpk
    z, p, k = tf2zpk(b, a)
    return {
        "tf": {"b": b, "a": a},
        "sos": tf2sos(b, a),
        "zpk": {"z": z, "p": p, "k": k},
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

def _coefficient_mode(payload):
    mode = str(payload.get("coefficient_mode", payload.get("mode", "tf"))).lower()
    if mode not in {"tf", "sos", "zpk"}:
        raise ValueError(f"Unsupported coefficient mode: {mode}")
    return mode

def _coefficients_from_payload(payload):
    from scipy.signal import sos2tf, zpk2tf
    mode = _coefficient_mode(payload)
    if mode == "tf":
        return _parse_coefficients(payload.get("b"), "b"), _parse_coefficients(payload.get("a"), "a")
    if mode == "sos":
        b, a = sos2tf(_parse_sos(payload.get("sos")))
        return _real_coefficients(b, "b"), _real_coefficients(a, "a")
    z, p, k = _parse_zpk(payload)
    b, a = zpk2tf(z, p, k)
    return _real_coefficients(b, "b"), _real_coefficients(a, "a")

def _parse_sos(value):
    import numpy as np
    if value is None or (isinstance(value, str) and value == ""):
        raise ValueError("Missing required field: sos")
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("Missing required field: sos")
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            import re
            value = [part for part in re.split(r"[\\s,]+", text) if part]
    sos = np.asarray(value, dtype=float)
    if sos.ndim == 1 and sos.size % 6 == 0:
        sos = sos.reshape((-1, 6))
    if sos.ndim != 2 or sos.shape[1] != 6 or sos.shape[0] == 0:
        raise ValueError("sos must have shape (n_sections, 6)")
    if not np.all(np.isfinite(sos)):
        raise ValueError("sos coefficients must be finite")
    return sos

def _parse_zpk(payload):
    import numpy as np
    raw = payload.get("zpk")
    if raw is not None:
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError("zpk must be valid JSON") from exc
        if isinstance(raw, (list, tuple)) and len(raw) == 3:
            z, p, k = raw
        elif isinstance(raw, dict):
            z = raw.get("z", raw.get("zeros"))
            p = raw.get("p", raw.get("poles"))
            k = raw.get("k", raw.get("gain", 1))
        else:
            raise ValueError("zpk must be an object or [z, p, k] array")
    else:
        z = payload.get("z", payload.get("zeros"))
        p = payload.get("p", payload.get("poles"))
        k = payload.get("k", payload.get("gain", 1))
    zeros = _parse_complex_array(z, "z")
    poles = _parse_complex_array(p, "p")
    gain = complex(_parse_complex_value(k, "k"))
    if not np.isfinite(gain.real) or not np.isfinite(gain.imag):
        raise ValueError("k must be finite")
    return zeros, poles, gain

def _parse_complex_array(value, name):
    import numpy as np
    if value is None or (isinstance(value, str) and value == ""):
        return np.array([], dtype=complex)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return np.array([], dtype=complex)
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            import re
            value = [part for part in re.split(r"[\\s,]+", text) if part]
    try:
        values = np.asarray([_parse_complex_value(item, name) for item in value], dtype=complex)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an array") from exc
    if values.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if not np.all(np.isfinite(values.real)) or not np.all(np.isfinite(values.imag)):
        raise ValueError(f"{name} values must be finite")
    return values

def _parse_complex_value(value, name):
    if isinstance(value, dict):
        real = value.get("real", value.get("re", 0))
        imag = value.get("imag", value.get("im", 0))
        return complex(float(real), float(imag))
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return complex(float(value[0]), float(value[1]))
    if isinstance(value, str):
        return complex(value.replace("i", "j").replace(" ", ""))
    try:
        return complex(float(value), 0.0)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} values must be finite numbers") from exc

def _real_coefficients(value, name):
    import numpy as np
    coefficients = np.real_if_close(np.asarray(value), tol=1000)
    coefficients = np.asarray(coefficients, dtype=float)
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
        "coefficients": _coefficient_representations(b, a),
        "response": _frequency_response(b, a, fs, response_points),
    }))

def py_infer(payload_json):
    payload = json.loads(payload_json)
    fs = float(payload.get("fs", 48000))
    response_points = _response_points(payload.get("response_points"))
    b, a = _coefficients_from_payload(payload)
    inferred = infer_iir_params(b, a, fs)
    return json.dumps(_json_safe({
        "inferred": inferred,
        "coefficients": _coefficient_representations(b, a),
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
    persistControlValues();
    setStatus("Pasted JSON");
    scheduleAutoDesign();
  } catch (error) {
    setStatus(error.message, "error");
  }
}

function resetDesign() {
  clearTimeout(autoDesignTimer);
  setFormValues(DEFAULT_DESIGN_PARAMS);
  designResponsePointsInput.value = DEFAULT_RESPONSE_POINTS;
  applyMethodDefaults();
  persistControlValues();
  scheduleAutoDesign();
}

function resetInference() {
  clearTimeout(autoInferTimer);
  setRadioValue(COEFFICIENT_MODE_NAMES.inference, DEFAULT_INFERENCE_MODE);
  applyInferenceModeDetails(DEFAULT_INFERENCE_MODE);
  inferenceCoeffTextarea.value = INFERENCE_MODE_DETAILS[DEFAULT_INFERENCE_MODE].sample;
  inferForm.elements.fs.value = String(DEFAULT_DESIGN_PARAMS.fs);
  inferenceResponsePointsInput.value = DEFAULT_RESPONSE_POINTS;
  lastInferenceMode = DEFAULT_INFERENCE_MODE;
  persistControlValues();
  scheduleAutoInfer();
}

function cssColor(name, fallback) {
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return value || fallback;
}

function chartThemeColors() {
  return {
    background: cssColor("--chart-bg", "#ffffff"),
    grid: cssColor("--chart-grid", "#dce4ea"),
    label: cssColor("--chart-label", "#66727c"),
    line: cssColor("--chart-line", "#172026"),
    frame: cssColor("--chart-frame", "#006d77"),
  };
}

function drawChart(canvas, response) {
  const frequencies = response?.frequency_hz || [];
  const magnitudes = response?.magnitude_db || [];
  const ctx = canvas.getContext("2d");
  const pixelRatio = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const colors = chartThemeColors();
  canvas.width = Math.max(320, Math.floor(rect.width * pixelRatio));
  canvas.height = Math.max(260, Math.floor(rect.height * pixelRatio));
  ctx.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);

  const width = rect.width;
  const height = rect.height;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = colors.background;
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

  ctx.strokeStyle = colors.grid;
  ctx.lineWidth = 1;
  ctx.fillStyle = colors.label;
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

  const xTickCount = plotWidth < 300 ? 2 : 4;
  for (let i = 0; i <= xTickCount; i += 1) {
    const x = padding.left + (plotWidth * i) / xTickCount;
    const value = minX + ((maxX - minX) * i) / xTickCount;
    ctx.beginPath();
    ctx.moveTo(x, padding.top);
    ctx.lineTo(x, height - padding.bottom);
    ctx.stroke();
    ctx.textAlign = i === 0 ? "left" : i === xTickCount ? "right" : "center";
    ctx.fillText(`${Math.round(value)} Hz`, x, height - 16);
  }
  ctx.textAlign = "left";

  ctx.strokeStyle = colors.line;
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

  ctx.strokeStyle = colors.frame;
  ctx.lineWidth = 2;
  ctx.strokeRect(padding.left, padding.top, plotWidth, plotHeight);
}

setThemeMode(preferredThemeMode());
themeControls.forEach((control) => {
  control.addEventListener("change", () => {
    if (control.checked) {
      applyThemeMode(control.value);
    }
  });
});

designForm.addEventListener("input", (event) => {
  if (event.target.name === "method") {
    applyMethodDefaults();
  }
  persistControlValues();
  scheduleAutoDesign();
});

designForm.addEventListener("change", (event) => {
  if (event.target.name === "method") {
    applyMethodDefaults();
  }
  persistControlValues();
  scheduleAutoDesign();
});

document.querySelector("#copy-json").addEventListener("click", async () => {
  const representation = selectedRepresentation(demoState.designCoefficients, selectedDesignCoefficientMode());
  if (!representation) {
    return;
  }
  await writeClipboardText(JSON.stringify(representation, null, 2));
  setStatus("Copied");
});

document.querySelector("#copy-text").addEventListener("click", async () => {
  if (!demoState.designCoefficients) {
    return;
  }
  await writeClipboardText(coefficientTextForMode(demoState.designCoefficients, selectedDesignCoefficientMode()));
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
document.querySelector("#reset-design").addEventListener("click", resetDesign);
document.querySelector("#reset-inference").addEventListener("click", resetInference);

designResponsePointsInput.addEventListener("input", () => {
  persistControlValues();
  scheduleAutoDesign();
});
designResponsePointsInput.addEventListener("change", () => {
  persistControlValues();
  scheduleAutoDesign();
});
inferForm.addEventListener("input", () => {
  persistControlValues();
  scheduleAutoInfer();
});
inferForm.addEventListener("change", () => {
  persistControlValues();
  scheduleAutoInfer();
});
inferenceResponsePointsInput.addEventListener("input", () => {
  persistControlValues();
  scheduleAutoInfer();
});
inferenceResponsePointsInput.addEventListener("change", () => {
  persistControlValues();
  scheduleAutoInfer();
});

document.querySelectorAll(`input[name="${COEFFICIENT_MODE_NAMES.design}"]`).forEach((control) => {
  control.addEventListener("change", renderDesignCoefficients);
});

document.querySelectorAll(`input[name="${COEFFICIENT_MODE_NAMES.inference}"]`).forEach((control) => {
  control.addEventListener("change", () => {
    updateInferenceModeUi(true);
    persistControlValues();
    scheduleAutoInfer();
  });
});

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
restoreControlValues();
updateInferenceModeUi(false);
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
