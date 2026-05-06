# IIR Filter 設計與反推工具

提供 IIR 濾波器的設計與頻率響應視覺化，同時可根據係數推回設計參數以驗證。

## 安裝需求
```bash
pip install -r requirements.txt
```

## 使用方法
```bash
make run
```

## Web 版
啟動本機 Flask 介面：
```bash
python web_app.py
```

如果系統 PATH 尚未設定 Python，可改用 Codex bundled Python：
```powershell
& "C:\Users\user\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" web_app.py
```

啟動後開啟 http://127.0.0.1:5000；若 5000 已被占用，程式會改用 http://127.0.0.1:5001。

## 功能
- 支援濾波器型態: lowpass, highpass, bandpass, notch
- 支援濾波器設計方法: Butterworth, Chebyshev, Elliptic, Bessel, RBJ Biquad
- 可反推濾波器參數以驗證係數來源
- 頻率響應圖繪製

## 套件結構
- `design.py`：濾波器設計
- `infer.py`：根據係數推估設計參數（目前為簡化實作）
- `plot.py`：畫頻率響應圖
- `example.py`：示範設計與反推驗證
- `web_app.py`：Flask Web 介面與 JSON API
