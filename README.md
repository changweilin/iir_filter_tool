# IIR Filter 設計與反推工具

提供 IIR 濾波器的設計與頻率響應視覺化，同時可根據係數推回設計參數以驗證。

## 安裝需求
```bash
pip install numpy scipy matplotlib
```

## 使用方法
```bash
make run
```

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