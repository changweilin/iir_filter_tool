
import numpy as np
from scipy.signal import butter, cheby1, cheby2, ellip, bessel

# --- 濾波器設計函式 ---
def design_iir(info, fs=None):
    """
    設計 IIR 濾波器，支援多種方法：
      - order=2 且 method='biquad'：RBJ Biquad
      - 其他：Butterworth, Cheby1, Cheby2, Elliptic, Bessel
    參數：
      info: dict 包含以下鍵值
        'ftype': 'lowpass','highpass','bandpass','notch'
        'f0'   : 截止/中心頻率 (Hz)
        'Q'    : 品質因數 (只於 biquad 有效，可為 None)
        'order': 階數
        'method': 'biquad','butterworth','cheby1','cheby2','elliptic','bessel'
        'rp'   : 通帶最大波紋 (dB)，Cheby1/Ellip 用
        'rs'   : 阻帶最小衰減 (dB)，Cheby2/Ellip 用
        'fs'   : 取樣率 (Hz，可選)
      fs  : 外部提供取樣率 (Hz)，優先於 info['fs']
    回傳：
      b, a : 濾波器係數
    """
    order = info['order']
    ftype = info['ftype']
    f0 = info['f0']
    Q = info.get('Q', None)
    method = info['method']
    rp = info.get('rp', None)
    rs = info.get('rs', None)

    # 取樣率優先採用函式參數，否則從 info 取值，預設為 48000
    fs = fs or info.get('fs', 48000)
    nyq = fs / 2.0
    m = method.lower()

    # RBJ Biquad (僅限二階)
    if order == 2 and m == 'biquad':
        w0 = 2 * np.pi * f0 / fs
        alpha = np.sin(w0) / (2 * Q)
        cosp = np.cos(w0)
        if ftype == 'lowpass':
            b0, b1, b2 = (1 - cosp) / 2, 1 - cosp, (1 - cosp) / 2
        elif ftype == 'highpass':
            b0, b1, b2 = (1 + cosp) / 2, -(1 + cosp), (1 + cosp) / 2
        elif ftype == 'bandpass':
            b0, b1, b2 = alpha, 0, -alpha
        elif ftype == 'notch':
            b0, b1, b2 = 1, -2 * cosp, 1
        else:
            raise ValueError(f"不支援的 ftype: {ftype}")
        a0 = 1 + alpha
        a1 = -2 * cosp
        a2 = 1 - alpha
        b = np.array([b0, b1, b2]) / a0
        a = np.array([1, a1 / a0, a2 / a0])
        return b, a

    # SciPy IIR 設計
    if ftype in ['lowpass', 'highpass']:
        Wn = f0 / nyq
    else:
        BW = (f0 / Q) if (Q and m in ['cheby1', 'cheby2', 'elliptic']) else (f0 * 0.1)
        f1 = max((f0 - BW / 2) / nyq, 1e-6)
        f2 = min((f0 + BW / 2) / nyq, 0.999)
        Wn = [f1, f2]

    if m in ['butterworth', 'butter']:
        b, a = butter(order, Wn, btype=ftype, analog=False)
    elif m == 'cheby1':
        b, a = cheby1(order, rp, Wn, btype=ftype, analog=False)
    elif m == 'cheby2':
        b, a = cheby2(order, rs, Wn, btype=ftype, analog=False)
    elif m in ['elliptic', 'ellip']:
        b, a = ellip(order, rp, rs, Wn, btype=ftype, analog=False)
    elif m == 'bessel':
        b, a = bessel(order, Wn, btype=ftype, analog=False)
    else:
        raise ValueError(f"不支援的設計方法：{method}")
    return b, a