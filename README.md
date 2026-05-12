# IIR Filter Tool

## 1) 專案標題與簡介 (Title & Description)

**IIR Filter Tool** 是一個以 Python 撰寫的 IIR 濾波器工具箱，提供：

- 可重複使用的濾波器係數設計函式（`design_iir`）
- 係數反推/解析資訊的分析函式（`infer_iir_params`）
- 幅度響應視覺化與快速驗證
- Flask Web UI（支援設計與反推）
- JSON API（`/api/design`、`/api/infer`）
- 可離線瀏覽的 GitHub Pages 靜態 Demo 產生流程

此專案定位為：
- 研究/教學可用的 IIR 濾波器設計範本
- 小型工具型專案（可直接用於實驗、驗證與導入到其他系統）

---

## 2) 核心功能特性 (Features)

### 核心演算法
- 支援 `ftype`：`lowpass`、`highpass`、`bandpass`、`notch`
- 設計方法（`method`）
  - `biquad`（RBJ biquad，僅限 `order=2`）
  - `butterworth`
  - `cheby1`
  - `cheby2`
  - `elliptic`
  - `bessel`
- 參數驗證與錯誤訊息友善：不合法欄位與範圍會明確回傳 400（API）或 `ValueError`（Python API）

### Python API
- `design_iir(params, fs=None)`：依參數產生 IIR 係數 (`b`, `a`)
- `infer_iir_params(b, a, fs)`：從係數推估 `ftype`、`f0`、`Q`、`method`、`order` 等 metadata
- `plot_response(b, a, fs, title)`：繪製幅度響應（Matplotlib）

### Web 介面與互動
- Flask 路由：
  - `GET /`：載入 Web UI
  - `POST /api/design`：回傳設計係數與頻率響應
  - `POST /api/infer`：輸入係數後回傳推估參數與頻率響應
- 支援三種係數輸入/輸出表示法
  - Transfer Function（`tf`）：`b`, `a`
  - Second-Order Section（`sos`）
  - Zero-Pole-Gain（`zpk`）
- 可調 `response_points`（2 ~ 65536）控制頻率取樣點數
- 回傳資料皆作 JSON-safe 格式處理（含複數欄位會輸出 `{"real", "imag"}`）

### GitHub Pages 靜態 Demo
- 透過 `scripts/build_pages_demo.py` 將主應用資料編譯為可部署靜態 `site/`
- 支援主題切換、響應式版面、係數複製／貼上、即時預覽波特圖

### 驗證與流程
- 包含 Python 單元測試：`tests/`
- CI 工作流程：
  - `Python and web tests`（`.github/workflows/ci.yml`）
  - `Pages`（`.github/workflows/pages.yml`）會建置並檢查靜態 demo

---

## 3) 系統需求與安裝步驟 (Prerequisites & Installation)

### 系統需求
- Python 3.8+（建議與 CI 一致：3.12）
- pip
- Windows / macOS / Linux 任一支援環境

### 建議安裝流程（PowerShell）

```powershell
# 建立虛擬環境
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 安裝相依套件
python -m pip install -r requirements.txt
```

或直接用 `py`：

```powershell
py -m pip install -r requirements.txt
```

### 安裝流程（Bash）

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### 安裝套件
`requirements.txt` 目前包含：

- `numpy`
- `scipy`
- `matplotlib`
- `flask`

---

## 4) 快速上手與使用範例 (Quick Start / Usage)

### 4.1 直接執行範例腳本

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
# 預設嘗試 5000，再嘗試 5001
python web_app.py
```

指定 PORT：

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

回傳包含：
- `b`, `a`
- `coefficients`（`tf`, `sos`, `zpk`）
- `response.frequency_hz`, `response.magnitude_db`

#### POST `/api/infer`

```bash
curl -X POST http://127.0.0.1:5000/api/infer \
  -H "Content-Type: application/json" \
  -d '{"b":[0.0127622136,0,-0.0127622136],"a":[1.0,-1.81534108,0.83100559],"fs":48000}'
```

回傳包含：
- `inferred.ftype`, `inferred.f0`, `inferred.Q`, `inferred.order`, `inferred.designable`
- `response.frequency_hz`, `response.magnitude_db`

### 4.5 建置並預覽 GitHub Pages 靜態 Demo

```bash
python scripts/build_pages_demo.py --output site
```

執行後可用靜態伺服器預覽：

```bash
python -m http.server --directory site 8080
```

在瀏覽器開啟 `http://127.0.0.1:8080`。

### 4.6 執行測試

```bash
python -m unittest discover -s tests
```

---

## 5) 專案架構說明 (Project Structure)

```text
iir_filter_tool/
├─ iir_filter/                  # Python 核心套件
│  ├─ __init__.py              # 匯出主要 API
│  ├─ design.py                # filter 設計邏輯（design_iir）
│  ├─ infer.py                 # 係數推估/解析（infer_iir_params）
│  └─ plot.py                  # 頻率響應繪圖（plot_response）
├─ web_app.py                   # Flask Web UI 與 JSON API
├─ example.py                   # CLI 示例
├─ scripts/
│  └─ build_pages_demo.py       # 產生 GitHub Pages 靜態 demo
├─ templates/
│  └─ index.html                # Flask HTML 模板
├─ static/
│  ├─ styles.css                # UI 樣式（含深淺色主題、RWD）
│  ├─ app.js                    # 前端邏輯（設計/推估/係數轉換/圖表）
│  └─ demo.js                   # 靜態 demo 專用前端邏輯
├─ tests/
│  ├─ test_iir_filter.py
│  ├─ test_web_app.py
│  └─ test_pages_demo.py
├─ docs/images/                 # README/展示截圖資源
├─ .github/workflows/
│  ├─ ci.yml                    # CI 測試與 Flask smoke test
│  └─ pages.yml                 # GitHub Pages build/deploy
├─ requirements.txt             # 相依套件
├─ Makefile                     # 便捷指令
└─ .gitignore
```

### Makefile 常用指令

```bash
make run         # 執行 example.py
make web         # 執行 web_app.py
make pages-demo  # 產生 site/ 靜態 demo
```

---

## 6) 授權條款 (License)

本專案採用 **MIT License**，授權條件如下：

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
