# IIR Filter Tool

一個用 Python 實作的 IIR 濾波器設計與分析工具，包含可匯入的 Python API、範例程式，以及 Flask Web UI。它可以設計常見 IIR 濾波器、畫出頻率響應，並從既有係數推估濾波器參數。

## 功能

- 支援濾波器類型：`lowpass`、`highpass`、`bandpass`、`notch`
- 支援設計方法：RBJ `biquad`、`butterworth`、`cheby1`、`cheby2`、`elliptic`、`bessel`
- 從 `b` / `a` 係數推估濾波器類型、截止或中心頻率、`Q`、階數、漣波、阻帶衰減、群延遲變化、零點與極點
- Web UI 可互動設計濾波器、檢視 magnitude response、複製 JSON 係數，並從係數反推參數
- 提供 JSON API，方便整合到其他工具或前端

## 安裝

建議先建立虛擬環境，再安裝相依套件。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

若你已經有可用的 Python 環境，也可以直接安裝：

```bash
pip install -r requirements.txt
```

如果 Windows 環境中 `python` 不在 `PATH`，可改用 Python launcher：

```powershell
py -m pip install -r requirements.txt
```

在 Codex bundled Python 環境中，也可以直接使用完整路徑：

```powershell
& "C:\Users\user\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m pip install -r requirements.txt
```

## Python 快速開始

執行內建範例：

```bash
python example.py
```

或在自己的程式中使用：

```python
from iir_filter import design_iir, infer_iir_params, plot_response

fs = 48000
params = {
    "ftype": "bandpass",
    "f0": 1000,
    "Q": 5,
    "order": 2,
    "method": "biquad",
}

b, a = design_iir(params, fs=fs)
print("b:", b)
print("a:", a)

inferred = infer_iir_params(b, a, fs)
print(inferred)

plot_response(b, a, fs=fs, title="Bandpass response")
```

## Web UI

啟動 Flask Web App：

```bash
python web_app.py
```

若 `python` 不在 `PATH`，請改用 `py web_app.py` 或上方的完整 Python 路徑。

預設會使用 [http://127.0.0.1:5000](http://127.0.0.1:5000)，如果 `5000` 已被占用，會自動嘗試 `5001`。

也可以指定埠號：

```powershell
$env:PORT = "5050"
python web_app.py
```

Makefile 也提供簡單指令：

```bash
make run
make web
make pages-demo
```

## GitHub Pages 靜態展示版

GitHub Pages 不能執行 Flask server，因此專案提供一個靜態展示頁產生器。它會輸出可直接發布的靜態檔案，並在瀏覽器中透過 Pyodide 載入 Python、NumPy、SciPy，直接執行與 Flask 版相同的濾波器設計與反推邏輯。第一次開頁需要下載 Pyodide/SciPy，載入完成後即可輸入參數並計算所有支援的方法。

```bash
python scripts/build_pages_demo.py --output site
```

產出的 `site/index.html` 可以直接由 GitHub Pages 發布。`.github/workflows/pages.yml` 會在推送到 `main` 或 `master` 時自動建立展示頁並部署到 `https://<owner>.github.io/<repo>/`。

## JSON API

### `POST /api/design`

根據設計參數回傳濾波器係數、頻率響應與反推參數。

Request:

```json
{
  "ftype": "bandpass",
  "method": "biquad",
  "fs": 48000,
  "f0": 1000,
  "Q": 5,
  "order": 2,
  "rp": null,
  "rs": null
}
```

Response 欄位包含：

- `b`: numerator coefficients
- `a`: denominator coefficients
- `response.frequency_hz`: 頻率座標
- `response.magnitude_db`: magnitude response，單位 dB
- `inferred`: 從係數推估出的分析結果

### `POST /api/infer`

根據既有 `b` / `a` 係數推估參數並回傳頻率響應。

Request:

```json
{
  "b": [0.01276221, 0, -0.01276221],
  "a": [1, -1.95676142, 0.97447558],
  "fs": 48000
}
```

`b` 和 `a` 可以是 JSON array，也可以是逗號或空白分隔的字串。

## 參數說明

- `fs`: 取樣率，必須為正數
- `f0`: 截止頻率或中心頻率，必須小於 Nyquist frequency，也就是 `fs / 2`
- `Q`: RBJ `biquad` 必填；`bandpass` / `notch` 設計時會用來決定頻帶寬度
- `order`: 濾波器階數；RBJ `biquad` 只能使用 `order=2`
- `rp`: Chebyshev I 與 Elliptic 設計所需的 passband ripple，單位 dB
- `rs`: Chebyshev II 與 Elliptic 設計所需的 stopband attenuation，單位 dB
- `norm`: Bessel 設計可用 `phase`、`delay`、`mag`，未指定時預設為 `phase`

## 注意事項

- `infer_iir_params` 是 best-effort 分析工具，不保證所有濾波器都能精準還原成原始設計參數。
- 目前 RBJ biquad 係數可較可靠地反推出可重新設計的參數；多數 SciPy prototype 濾波器會回傳 analysis-only metadata。
- `notch` 在 SciPy-backed 設計中會映射為 `bandstop`。

## 測試

```bash
python -m unittest discover -s tests
```

## 專案結構

```text
iir_filter/
  design.py      # IIR 設計邏輯
  infer.py       # 係數分析與參數推估
  plot.py        # Matplotlib 頻率響應繪圖
example.py       # Python API 使用範例
web_app.py       # Flask Web UI 與 JSON API
scripts/         # GitHub Pages 靜態展示頁產生器
templates/       # HTML template
static/          # 前端 JavaScript 與 CSS
tests/           # 單元測試
```
