const state = {
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
  return {
    b: data.get("b"),
    a: data.get("a"),
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

function renderResult(data) {
  if (data.b && data.a) {
    state.coefficients = { b: data.b, a: data.a };
    renderList("#b-list", data.b);
    renderList("#a-list", data.a);
    document.querySelector("#coeff-json").textContent = JSON.stringify(state.coefficients, null, 2);
    inferForm.elements.b.value = JSON.stringify(data.b);
    inferForm.elements.a.value = JSON.stringify(data.a);
    inferForm.elements.fs.value = designForm.elements.fs.value;
  }

  if (data.inferred) {
    state.inferred = data.inferred;
    renderSummary(data.inferred);
    document.querySelector("#inferred-json").textContent = JSON.stringify(data.inferred, null, 2);
    applyInferredButton.disabled = !data.inferred.designable;
  }

  if (data.response) {
    state.response = data.response;
    drawChart(data.response);
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

async function runDesign() {
  try {
    const data = await postJson("/api/design", designPayload());
    renderResult(data);
  } catch (error) {
    setStatus(error.message, "error");
  }
}

async function runInfer() {
  try {
    const data = await postJson("/api/infer", inferPayload());
    renderResult(data);
  } catch (error) {
    setStatus(error.message, "error");
  }
}

function applyInferred() {
  const inferred = state.inferred;
  if (!inferred || !inferred.designable) {
    return;
  }
  ["ftype", "method", "fs", "f0", "order"].forEach((field) => {
    if (designForm.elements[field] && inferred[field] != null) {
      designForm.elements[field].value = inferred[field];
    }
  });

  designForm.elements.Q.value = Number(inferred.Q) > 0 ? inferred.Q : "";
  designForm.elements.rp.value = "";
  designForm.elements.rs.value = "";

  if (["cheby1", "elliptic"].includes(inferred.method) && Number(inferred.rp) > 0) {
    designForm.elements.rp.value = inferred.rp;
  }
  if (["cheby2", "elliptic"].includes(inferred.method) && Number(inferred.rs) > 0) {
    designForm.elements.rs.value = inferred.rs;
  }
}

document.querySelector("#design-submit").addEventListener("click", runDesign);
document.querySelector("#infer-submit").addEventListener("click", runInfer);
document.querySelector("#copy-json").addEventListener("click", async () => {
  if (!state.coefficients) {
    return;
  }
  await navigator.clipboard.writeText(JSON.stringify(state.coefficients, null, 2));
  setStatus("Copied");
});
applyInferredButton.addEventListener("click", applyInferred);
window.addEventListener("resize", () => {
  if (state.response) {
    drawChart(state.response);
  }
});

runDesign();
