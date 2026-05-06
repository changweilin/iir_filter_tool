const state = {
  design: {
    coefficients: null,
    response: null,
  },
  inference: {
    inferred: null,
    response: null,
  },
};

const statusEl = document.querySelector("#status");
const designChart = document.querySelector("#response-chart");
const designChartMeta = document.querySelector("#chart-meta");
const inferenceChart = document.querySelector("#inference-response-chart");
const inferenceChartMeta = document.querySelector("#inference-chart-meta");
const designForm = document.querySelector("#design-form");
const inferForm = document.querySelector("#infer-form");
let autoInferTimer = null;
let lastCopiedInferredJson = "";

function setStatus(message, mode = "ready") {
  statusEl.textContent = message;
  statusEl.classList.toggle("is-error", mode === "error");
  statusEl.classList.toggle("is-working", mode === "working");
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
  };
}

function inferPayload() {
  const data = new FormData(inferForm);
  const coefficients = parseCoefficientText(data.get("coefficients"));
  return {
    b: coefficients.b,
    a: coefficients.a,
    fs: numberOrNull(data.get("fs")),
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
    setStatus("Pasted JSON");
    await runDesign();
  } catch (error) {
    setStatus(error.message, "error");
  }
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

function renderDesignResult(data) {
  if (data.b && data.a) {
    state.design.coefficients = { b: data.b, a: data.a };
    renderList("#b-list", data.b);
    renderList("#a-list", data.a);
    document.querySelector("#coeff-json").textContent = JSON.stringify(state.design.coefficients, null, 2);
  }

  if (data.response) {
    state.design.response = data.response;
    drawChart(designChart, designChartMeta, data.response);
  }
}

function renderInferenceResult(data) {
  if (data.inferred) {
    state.inference.inferred = data.inferred;
    renderSummary(data.inferred);
    document.querySelector("#inferred-json").textContent = JSON.stringify(data.inferred, null, 2);
  }

  if (data.response) {
    state.inference.response = data.response;
    drawChart(inferenceChart, inferenceChartMeta, data.response);
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

document.querySelector("#design-submit").addEventListener("click", runDesign);
document.querySelector("#paste-design-json").addEventListener("click", pasteDesignJson);
document.querySelector("#copy-json").addEventListener("click", async () => {
  if (!state.design.coefficients) {
    return;
  }
  await writeClipboardText(JSON.stringify(state.design.coefficients, null, 2));
  setStatus("Copied");
});
document.querySelector("#copy-text").addEventListener("click", async () => {
  if (!state.design.coefficients) {
    return;
  }
  await writeClipboardText(coefficientText(state.design.coefficients));
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
window.addEventListener("resize", () => {
  if (state.design.response) {
    drawChart(designChart, designChartMeta, state.design.response);
  }
  if (state.inference.response) {
    drawChart(inferenceChart, inferenceChartMeta, state.inference.response);
  }
});

drawChart(inferenceChart, inferenceChartMeta, null);
runDesign();
scheduleAutoInfer();
