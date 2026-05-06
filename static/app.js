const state = {
  design: {
    coefficients: null,
    response: null,
  },
  inference: {
    inferred: null,
    coefficients: null,
    response: null,
  },
};

const FORM_STORAGE_KEY = "iir-filter-tool:form-state:v2";
const THEME_STORAGE_KEY = "iir-filter-tool:theme";
const THEME_MODES = new Set(["day", "night"]);
const COEFFICIENT_MODE_NAMES = {
  design: "design-coefficient-mode",
  inference: "inference-coefficient-mode",
};
const INFERENCE_MODE_DETAILS = {
  tf: {
    label: "Coefficient text (b0,b1,b2,a0,a1,a2)",
    placeholder: "b0,b1,b2,a0,a1,a2",
    sample: "[0.01276221, 0, -0.01276221, 1, -1.95676142, 0.97447558]",
  },
  sos: {
    label: "Second-order sections (JSON rows)",
    placeholder: "[[b0,b1,b2,a0,a1,a2]]",
    sample: "[[0.01276221, 0, -0.01276221, 1, -1.95676142, 0.97447558]]",
  },
  zpk: {
    label: "Zeros / poles / gain (JSON)",
    placeholder: '{"z":[1,-1],"p":[{"real":0.97838071,"imag":0.13132694},{"real":0.97838071,"imag":-0.13132694}],"k":0.01276221}',
    sample: '{"z":[1,-1],"p":[{"real":0.97838071,"imag":0.13132694},{"real":0.97838071,"imag":-0.13132694}],"k":0.01276221}',
  },
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
let autoInferTimer = null;
let lastCopiedInferredJson = "";
let lastInferenceMode = "tf";

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
  if (state.design.response) {
    drawChart(designChart, state.design.response);
  }
  drawChart(inferenceChart, state.inference.response);
}

function applyThemeMode(mode) {
  const theme = setThemeMode(mode);
  persistThemeMode(theme);
  redrawChartsForTheme();
}

function numberOrNull(value) {
  if (value === "" || value == null) {
    return null;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function designPayload() {
  const data = new FormData(designForm);
  return {
    ftype: data.get("ftype"),
    method: data.get("method"),
    fs: numberOrNull(data.get("fs")),
    f0: numberOrNull(data.get("f0")),
    Q: numberOrNull(data.get("Q")),
    order: numberOrNull(data.get("order")),
    rp: numberOrNull(data.get("rp")),
    rs: numberOrNull(data.get("rs")),
    response_points: responsePoints(designResponsePointsInput.value),
  };
}

function inferPayload() {
  const data = new FormData(inferForm);
  const coefficients = parseCoefficientInput(data.get("coefficients"), selectedInferenceCoefficientMode());
  return {
    ...coefficients,
    fs: numberOrNull(data.get("fs")),
    response_points: responsePoints(inferenceResponsePointsInput.value),
  };
}

async function postJson(url, payload) {
  setStatus("Working", "working");
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "Request failed");
  }
  setStatus("Ready");
  return data;
}

function formatNumber(value) {
  if (value == null) {
    return "null";
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

function responsePoints(value) {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < 2 || parsed > 65536) {
    throw new Error("Response points must be an integer between 2 and 65536");
  }
  return parsed;
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

async function pasteDesignJson() {
  try {
    const text = await readClipboardText(lastCopiedInferredJson);
    const parsed = JSON.parse(text);
    applyDesignParameters(parsed.inferred || parsed);
    persistControlValues();
    setStatus("Pasted JSON");
    await runDesign();
  } catch (error) {
    setStatus(error.message, "error");
  }
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
  const representation = selectedRepresentation(state.design.coefficients, mode);
  renderCoefficientView(designCoefficientView, state.design.coefficients, mode);
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
  const details = INFERENCE_MODE_DETAILS[mode];
  const previousDetails = INFERENCE_MODE_DETAILS[lastInferenceMode];
  const currentText = inferenceCoeffTextarea.value.trim();
  inferenceCoeffLabel.textContent = details.label;
  inferenceCoeffTextarea.placeholder = details.placeholder;

  if (useCurrentCoefficients && state.inference.coefficients?.[mode]) {
    inferenceCoeffTextarea.value = coefficientInputText(mode, state.inference.coefficients);
  } else if (!currentText || currentText === previousDetails.sample) {
    inferenceCoeffTextarea.value = details.sample;
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

function renderDesignResult(data) {
  const coefficients = coefficientRepresentations(data);
  if (coefficients) {
    state.design.coefficients = coefficients;
    renderDesignCoefficients();
  }

  if (data.response) {
    state.design.response = data.response;
    drawChart(designChart, data.response);
  }
}

function renderInferenceResult(data) {
  const coefficients = coefficientRepresentations(data);
  if (coefficients) {
    state.inference.coefficients = coefficients;
  }

  if (data.inferred) {
    state.inference.inferred = data.inferred;
    renderSummary(data.inferred);
    document.querySelector("#inferred-json").textContent = JSON.stringify(data.inferred, null, 2);
  }

  if (data.response) {
    state.inference.response = data.response;
    drawChart(inferenceChart, data.response);
  }
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

async function runDesign() {
  try {
    const data = await postJson("/api/design", designPayload());
    renderDesignResult(data);
  } catch (error) {
    setStatus(error.message, "error");
  }
}

async function runInfer() {
  try {
    const data = await postJson("/api/infer", inferPayload());
    renderInferenceResult(data);
  } catch (error) {
    setStatus(error.message, "error");
  }
}

function scheduleAutoInfer() {
  clearTimeout(autoInferTimer);
  autoInferTimer = setTimeout(runInfer, 450);
}

setThemeMode(preferredThemeMode());
themeControls.forEach((control) => {
  control.addEventListener("change", () => {
    if (control.checked) {
      applyThemeMode(control.value);
    }
  });
});

document.querySelector("#design-submit").addEventListener("click", runDesign);
document.querySelector("#paste-design-json").addEventListener("click", pasteDesignJson);
designForm.addEventListener("input", persistControlValues);
designForm.addEventListener("change", persistControlValues);
designResponsePointsInput.addEventListener("input", persistControlValues);
designResponsePointsInput.addEventListener("change", () => {
  persistControlValues();
  runDesign();
});
document.querySelector("#copy-json").addEventListener("click", async () => {
  const representation = selectedRepresentation(state.design.coefficients, selectedDesignCoefficientMode());
  if (!representation) {
    return;
  }
  await writeClipboardText(JSON.stringify(representation, null, 2));
  setStatus("Copied");
});
document.querySelector("#copy-text").addEventListener("click", async () => {
  if (!state.design.coefficients) {
    return;
  }
  await writeClipboardText(coefficientTextForMode(state.design.coefficients, selectedDesignCoefficientMode()));
  setStatus("Copied text");
});
document.querySelector("#copy-inferred-json").addEventListener("click", async () => {
  if (!state.inference.inferred) {
    return;
  }
  lastCopiedInferredJson = JSON.stringify(state.inference.inferred, null, 2);
  await writeClipboardText(lastCopiedInferredJson);
  setStatus("Copied inferred JSON");
});
inferForm.addEventListener("input", scheduleAutoInfer);
inferForm.addEventListener("change", scheduleAutoInfer);
inferForm.addEventListener("input", persistControlValues);
inferForm.addEventListener("change", persistControlValues);
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
  if (state.design.response) {
    drawChart(designChart, state.design.response);
  }
  if (state.inference.response) {
    drawChart(inferenceChart, state.inference.response);
  }
});

drawChart(inferenceChart, null);
restoreControlValues();
updateInferenceModeUi(false);
runDesign();
scheduleAutoInfer();
