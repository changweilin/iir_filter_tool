# IIR Filter Tool (IIR 濾波器工具)

## 1) 專案標案與簡介 / Title & Description

**中文**

`IIR Filter Tool` 是一個 Python 專案，提供 IIR 濾波器的設計、係數反推分析、回應繪圖與 Web 介面/JSON API 服務。你可以使用 RBJ biquad 或 SciPy 的常見設計方法（Butterworth、Chebyshev、Elliptic、Bessel）產生 IIR 係數，並用 API 將結果與幅度響應輸出給前端/腳本使用。

**English**

`IIR Filter Tool` is a Python project for IIR filter design, coefficient inference, response plotting, and serving results through a Flask web UI and JSON API. It supports RBJ biquad and common SciPy-based designs (Butterworth, Chebyshev, Elliptic, Bessel), and provides coefficients plus frequency-response data for downstream scripts or clients.

---

## 2) 核心功能特性 / Features

**中文**

- 設計類型：`lowpass`、`highpass`、`bandpass`、`notch`
- 設計方法：`biquad`（RBJ）與 SciPy 後端 `butterworth`、`cheby1`、`cheby2`、`elliptic`、`bessel`
- 支援參數驗證與錯誤回報（錯誤參數會回傳清楚訊息）
- Python API：`design_iir` / `infer_iir_params` / `plot_response`
- Web UI：可即時設計與繪圖、複製係數、主題切換、RWD 介面
- JSON API：
  - `POST /api/design`
  - `POST /api/infer`
- 靜態 Demo：可產生可部署到 GitHub Pages 的 `site/`
- 測試完整：核心邏輯、API、靜態 Demo 都有測試覆蓋

**English**

- Filter types: `lowpass`, `highpass`, `bandpass`, `notch`
- Design methods: `biquad` (RBJ) and SciPy-based `butterworth`, `cheby1`, `cheby2`, `elliptic`, `bessel`
- Input validation and clear error handling (invalid inputs return explicit error messages)
- Python API: `design_iir`, `infer_iir_params`, `plot_response`
- Flask web UI for interactive design, plotting, coefficient copy, theme switching, responsive layout
- JSON API endpoints:
  - `POST /api/design`
  - `POST /api/infer`
- Static demo generation to `site/` for GitHub Pages
- Tests for core logic, web app routes, and demo generation

---

## 3) 系統需求與安裝步驟 / Prerequisites & Installation

**中文**

### 系統需求
- Python 3.8+（建議 3.12）
- pip
- Windows / macOS / Linux

### 依賴套件
`requirements.txt` 目前包含：

- `numpy`
- `scipy`
- `matplotlib`
- `flask`

### 安裝（PowerShell）

```powershell
# 建立虛擬環境
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 安裝套件
python -m pip install -r requirements.txt
```

或使用 `py`：

```powershell
py -m pip install -r requirements.txt
```

### Install (Bash)

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

**English**

### Requirements
- Python 3.8+ (Python 3.12 recommended)
- `pip`
- Windows / macOS / Linux

### Dependencies
From `requirements.txt`:

- `numpy`
- `scipy`
- `matplotlib`
- `flask`

### Install (PowerShell)

```powershell
# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install -r requirements.txt
```

or with `py`:

```powershell
py -m pip install -r requirements.txt
```

### Install (Bash)

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

---

## 4) 快速上手與使用範例 / Quick Start / Usage

**中文**

### 4.1 執行範例腳本

```bash
python example.py
```

### 4.2 使用 Python API

```python
from iir_filter import design_iir, infer_iir_params, plot_response

fs = 48000
params = {
    "ftype": "bandpass",
    "method": "biquad",
    "f0": 1000,
    "Q": 5,
    "order": 2,
}

b, a = design_iir(params, fs=fs)
print("b:", b)
print("a:", a)

inferred = infer_iir_params(b, a, fs)
print(inferred)

plot_response(b, a, fs=fs, title="Bandpass response")
```

### 4.3 啟動 Flask Web UI

```bash
python web_app.py
```

預設會嘗試 `5000`，不可用再退回 `5001`。

指定連接埠：

```powershell
$env:PORT = "5050"
python web_app.py
```

### 4.4 呼叫 JSON API

#### POST `/api/design`

```bash
curl -X POST http://127.0.0.1:5000/api/design \
  -H "Content-Type: application/json" \
  -d '{"ftype":"bandpass","method":"biquad","fs":48000,"f0":1000,"Q":5,"order":2}'
```

#### POST `/api/infer`

```bash
curl -X POST http://127.0.0.1:5000/api/infer \
  -H "Content-Type: application/json" \
  -d '{"b":[0.0127622136,0,-0.0127622136],"a":[1.0,-1.81534108,0.83100559],"fs":48000}'
```

### 4.5 產生 GitHub Pages 靜態 Demo

```bash
python scripts/build_pages_demo.py --output site
python -m http.server --directory site 8080
```

**English**

### 4.1 Run the example script

```bash
python example.py
```

### 4.2 Use Python API

```python
from iir_filter import design_iir, infer_iir_params, plot_response

fs = 48000
params = {
    "ftype": "bandpass",
    "method": "biquad",
    "f0": 1000,
    "Q": 5,
    "order": 2,
}

b, a = design_iir(params, fs=fs)
print("b:", b)
print("a:", a)

inferred = infer_iir_params(b, a, fs)
print(inferred)

plot_response(b, a, fs=fs, title="Bandpass response")
```

### 4.3 Run Flask Web UI

```bash
python web_app.py
```

By default it tries port `5000`, then falls back to `5001` if needed.

Set a fixed port:

```bash
export FLASK_APP=web_app: create_app
# (optional) when running on Windows PowerShell:
$env:PORT = "5050"
python web_app.py
```

### 4.4 Call JSON API

#### POST `/api/design`

```bash
curl -X POST http://127.0.0.1:5000/api/design \
  -H "Content-Type: application/json" \
  -d '{"ftype":"bandpass","method":"biquad","fs":48000,"f0":1000,"Q":5,"order":2}'
```

#### POST `/api/infer`

```bash
curl -X POST http://127.0.0.1:5000/api/infer \
  -H "Content-Type: application/json" \
  -d '{"b":[0.0127622136,0,-0.0127622136],"a":[1.0,-1.81534108,0.83100559],"fs":48000}'
```

### 4.5 Build and preview static Pages demo

```bash
python scripts/build_pages_demo.py --output site
python -m http.server --directory site 8080
```

Open `http://127.0.0.1:8080` in your browser.

### 4.6 執行測試 / Run tests

```bash
python -m unittest discover -s tests
```

---

## 5) 專案架構說明 / Project Structure

```text
iir_filter_tool/
├─ iir_filter/
│  ├─ __init__.py
│  ├─ design.py
│  ├─ infer.py
│  └─ plot.py
├─ web_app.py
├─ example.py
├─ scripts/
│  └─ build_pages_demo.py
├─ templates/
│  └─ index.html
├─ static/
│  ├─ styles.css
│  └─ app.js
│  └─ demo.js
├─ tests/
│  ├─ test_iir_filter.py
│  ├─ test_web_app.py
│  └─ test_pages_demo.py
├─ docs/images/
├─ .github/workflows/
│  ├─ ci.yml
│  └─ pages.yml
├─ requirements.txt
├─ Makefile
└─ README.md
```

### Makefile 常用指令 / Makefile shortcuts

```bash
make run        # python3 example.py
make web        # python3 web_app.py
make pages-demo # python3 scripts/build_pages_demo.py --output site
```

**English**

### Main layout

- `iir_filter/` contains the core algorithm modules.
- `web_app.py` exposes Flask UI and API endpoints.
- `scripts/build_pages_demo.py` generates static output for GitHub Pages.
- `templates/` and `static/` contain the web UI.
- `tests/` includes unit tests for filter logic, web API, and demo generation.
- `.github/workflows/` contains CI and Pages pipelines.

---

## 6) 授權條款 / License

本專案採用 **MIT License**。

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
