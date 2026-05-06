const demoCases = JSON.parse(document.querySelector("#demo-data").textContent);
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

function renderCase(index) {
  demoState.index = index;
  const demoCase = demoCases[index];
  const params = demoCase.params;

  ["ftype", "method", "fs", "f0", "Q", "order", "rp", "rs"].forEach((field) => {
    if (designForm.elements[field]) {
      designForm.elements[field].value = formatNumber(params[field]);
    }
  });

  demoState.coefficients = { b: demoCase.b, a: demoCase.a };
  demoState.inferred = demoCase.inferred;
  demoState.response = demoCase.response;

  renderList("#b-list", demoCase.b);
  renderList("#a-list", demoCase.a);
  renderSummary(demoCase.inferred);
  drawChart(demoCase.response);

  inferForm.elements.b.value = JSON.stringify(demoCase.b);
  inferForm.elements.a.value = JSON.stringify(demoCase.a);
  inferForm.elements.fs.value = params.fs;
  document.querySelector("#coeff-json").textContent = JSON.stringify(demoState.coefficients, null, 2);
  document.querySelector("#inferred-json").textContent = JSON.stringify(demoCase.inferred, null, 2);
  setStatus(demoCase.title);

  [...presetList.children].forEach((button, buttonIndex) => {
    button.classList.toggle("is-active", buttonIndex === index);
  });
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

document.querySelector("#design-submit").addEventListener("click", () => {
  renderCase((demoState.index + 1) % demoCases.length);
});

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
  ["ftype", "method", "fs", "f0", "Q", "order", "rp", "rs"].forEach((field) => {
    if (designForm.elements[field]) {
      designForm.elements[field].value = formatNumber(inferred[field]);
    }
  });
  setStatus("Applied");
});

window.addEventListener("resize", () => {
  if (demoState.response) {
    drawChart(demoState.response);
  }
});

renderPresetButtons();
renderCase(0);
