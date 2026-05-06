const demoCases = JSON.parse(document.querySelector("#demo-data").textContent);
const RESPONSE_POINTS = 1024;

const demoState = {
  index: 0,
  coefficients: null,
  inferred: null,
  response: null,
};

const statusEl = document.querySelector("#status");
const chart = document.querySelector("#response-chart");
const chartMeta = document.querySelector("#chart-meta");
const designForm = document.querySelector("#design-form");
const inferForm = document.querySelector("#infer-form");
const applyInferredButton = document.querySelector("#apply-inferred");
const presetList = document.querySelector("#preset-list");

function setStatus(message, mode = "ready") {
  statusEl.textContent = message;
  statusEl.classList.toggle("is-error", mode === "error");
  statusEl.classList.toggle("is-working", mode === "working");
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
    Q: positiveNumber(data.get("Q"), "Q"),
    order: positiveInteger(data.get("order"), "order"),
    rp: numberOrNull(data.get("rp")),
    rs: numberOrNull(data.get("rs")),
  };
}

function renderCase(index) {
  demoState.index = index;
  const demoCase = demoCases[index];
  setFormValues(demoCase.params);
  renderResult({
    b: demoCase.b,
    a: demoCase.a,
    response: demoCase.response,
    inferred: demoCase.inferred,
  });
  setStatus(demoCase.title);

  [...presetList.children].forEach((button, buttonIndex) => {
    button.classList.toggle("is-active", buttonIndex === index);
  });
}

function renderResult(data) {
  demoState.coefficients = { b: data.b, a: data.a };
  demoState.inferred = data.inferred;
  demoState.response = data.response;

  renderList("#b-list", data.b);
  renderList("#a-list", data.a);
  renderSummary(data.inferred);
  drawChart(data.response);

  inferForm.elements.b.value = JSON.stringify(data.b);
  inferForm.elements.a.value = JSON.stringify(data.a);
  inferForm.elements.fs.value = designForm.elements.fs.value;
  document.querySelector("#coeff-json").textContent = JSON.stringify(demoState.coefficients, null, 2);
  document.querySelector("#inferred-json").textContent = JSON.stringify(data.inferred, null, 2);
}

function designBiquad(params) {
  if (params.method !== "biquad") {
    throw new Error("Static custom design supports biquad only. Use presets for SciPy methods.");
  }
  if (params.order !== 2) {
    throw new Error("RBJ biquad design requires order=2");
  }
  if (params.f0 >= params.fs / 2) {
    throw new Error("f0 must be below the Nyquist frequency");
  }

  const w0 = (2 * Math.PI * params.f0) / params.fs;
  const alpha = Math.sin(w0) / (2 * params.Q);
  const cosp = Math.cos(w0);
  let b0;
  let b1;
  let b2;

  if (params.ftype === "lowpass") {
    b0 = (1 - cosp) / 2;
    b1 = 1 - cosp;
    b2 = (1 - cosp) / 2;
  } else if (params.ftype === "highpass") {
    b0 = (1 + cosp) / 2;
    b1 = -(1 + cosp);
    b2 = (1 + cosp) / 2;
  } else if (params.ftype === "bandpass") {
    b0 = alpha;
    b1 = 0;
    b2 = -alpha;
  } else if (params.ftype === "notch") {
    b0 = 1;
    b1 = -2 * cosp;
    b2 = 1;
  } else {
    throw new Error(`Unsupported filter type: ${params.ftype}`);
  }

  const a0 = 1 + alpha;
  const a1 = -2 * cosp;
  const a2 = 1 - alpha;
  return {
    b: [b0 / a0, b1 / a0, b2 / a0],
    a: [1, a1 / a0, a2 / a0],
  };
}

function frequencyResponse(b, a, fs) {
  const frequencyHz = [];
  const magnitudeDb = [];
  for (let i = 0; i < RESPONSE_POINTS; i += 1) {
    const frequency = (i * fs) / (2 * RESPONSE_POINTS);
    const w = (2 * Math.PI * frequency) / fs;
    const numerator = evaluatePolynomial(b, w);
    const denominator = evaluatePolynomial(a, w);
    const denomPower = denominator.real * denominator.real + denominator.imag * denominator.imag;
    const real = (numerator.real * denominator.real + numerator.imag * denominator.imag) / denomPower;
    const imag = (numerator.imag * denominator.real - numerator.real * denominator.imag) / denomPower;
    const magnitude = Math.max(Math.hypot(real, imag), Number.MIN_VALUE);
    frequencyHz.push(frequency);
    magnitudeDb.push(20 * Math.log10(magnitude));
  }
  return {
    frequency_hz: frequencyHz,
    magnitude_db: magnitudeDb,
  };
}

function evaluatePolynomial(coefficients, w) {
  return coefficients.reduce(
    (sum, coefficient, index) => {
      const angle = -w * index;
      return {
        real: sum.real + coefficient * Math.cos(angle),
        imag: sum.imag + coefficient * Math.sin(angle),
      };
    },
    { real: 0, imag: 0 },
  );
}

function inferBiquad(params, b, a) {
  return {
    ftype: params.ftype,
    f0: params.f0,
    Q: params.Q,
    order: Math.max(a.length - 1, 0),
    fs: params.fs,
    rp: params.rp,
    rs: params.rs,
    gd_dev: null,
    poles: [],
    zeros: [],
    method: "biquad",
    designable: true,
  };
}

function runDesign() {
  try {
    const params = currentDesignParams();
    const { b, a } = designBiquad(params);
    renderResult({
      b,
      a,
      response: frequencyResponse(b, a, params.fs),
      inferred: inferBiquad(params, b, a),
    });
    setStatus("Designed");
    [...presetList.children].forEach((button) => button.classList.remove("is-active"));
  } catch (error) {
    setStatus(error.message, "error");
  }
}

function drawChart(response) {
  const frequencies = response.frequency_hz || [];
  const magnitudes = response.magnitude_db || [];
  const ctx = chart.getContext("2d");
  const pixelRatio = window.devicePixelRatio || 1;
  const rect = chart.getBoundingClientRect();
  chart.width = Math.max(320, Math.floor(rect.width * pixelRatio));
  chart.height = Math.max(260, Math.floor(rect.height * pixelRatio));
  ctx.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);

  const width = rect.width;
  const height = rect.height;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);

  if (!frequencies.length || !magnitudes.length) {
    chartMeta.textContent = "0 points";
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
  chartMeta.textContent = `${frequencies.length} points`;
}

document.querySelector("#design-submit").addEventListener("click", runDesign);

document.querySelector("#infer-submit").addEventListener("click", () => {
  document.querySelector("#inferred-json").textContent = JSON.stringify(demoState.inferred, null, 2);
  setStatus("Inferred");
});

document.querySelector("#copy-json").addEventListener("click", async () => {
  if (!demoState.coefficients) {
    return;
  }
  await navigator.clipboard.writeText(JSON.stringify(demoState.coefficients, null, 2));
  setStatus("Copied");
});

applyInferredButton.addEventListener("click", () => {
  const inferred = demoState.inferred;
  if (!inferred) {
    return;
  }
  setFormValues(inferred);
  setStatus("Applied");
});

window.addEventListener("resize", () => {
  if (demoState.response) {
    drawChart(demoState.response);
  }
});

renderPresetButtons();
renderCase(0);
