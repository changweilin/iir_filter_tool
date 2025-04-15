# %% [markdown]
# 根據推測參數重建濾波器或反向推測濾波器設計參數，並交互驗證

# %% [code]
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz, butter, cheby1, cheby2, ellip, bessel, tf2zpk, group_delay, find_peaks



# --- 濾波器分析與推測函式 ---
def analyze_iir(b, a, fs, worN=8000):
    """
    分析任意階 IIR 濾波器，回傳 f, mag, ftype, f0, Q
    """
    f, h = freqz(b, a, worN=worN, fs=fs)
    mag = 20 * np.log10(np.abs(h))
    mag0, magNy = mag[0], mag[-1]
    idx_peak, idx_valley = np.argmax(mag), np.argmin(mag)
    f_peak, f_valley = f[idx_peak], f[idx_valley]

    # 判斷類型與 f0, Q
    if idx_peak == 0:
        ftype = 'lowpass'
        thr = mag0 - 3
        idx_cut = np.where(mag <= thr)[0][0]
        f0, Q = f[idx_cut], None

    elif idx_peak == len(f)-1:
        ftype = 'highpass'
        thr = magNy - 3
        idx_cut = np.where(mag >= thr)[0][0]
        f0, Q = f[idx_cut], None

    elif 0 < idx_peak < len(f)-1 and mag[idx_peak] > mag0 and mag[idx_peak] > magNy:
        ftype = 'bandpass'
        f0 = f_peak
        thr = mag[idx_peak] - 3
        idx1 = np.where(mag[:idx_peak] <= thr)[0][-1]
        idx2 = idx_peak + np.where(mag[idx_peak:] <= thr)[0][0]
        f1, f2 = f[idx1], f[idx2]
        Q = f0 / (f2 - f1)

    elif 0 < idx_valley < len(f)-1 and mag[idx_valley] < mag0 and mag[idx_valley] < magNy:
        ftype = 'notch'
        f0 = f_valley
        thr = mag[idx_valley] + 3
        idx1 = np.where(mag[:idx_valley] >= thr)[0][-1]
        idx2 = idx_valley + np.where(mag[idx_valley:] >= thr)[0][0]
        f1, f2 = f[idx1], f[idx2]
        Q = f0 / (f2 - f1)

    else:
        ftype, f0, Q = 'unknown', None, None

    return f, mag, ftype, f0, Q


def infer_iir_params(b, a, fs):
    """
    自動從 b, a 推測：
      - 濾波器類型 ftype
      - 截止／中心頻率 f0
      - 品質因數 Q (若有)
      - 通帶波紋 rp (dB)
      - 阻帶衰減 rs (dB)
      - 群延遲變化量 gd_dev (samples)
      - 極點 poles, 零點 zeros
      - 最可能的設計方法 method
    """
    # 1. 先分析 ftype, f0, Q
    f, mag, ftype, f0, Q = analyze_iir(b, a, fs)

    # 2. 計算 rp, rs
    if ftype == 'lowpass':
        #find pass band
        pb = mag[f <= f0]
        pb_peak, _ = find_peaks(pb)
        if pb_peak.size > 0 and pb_peak[-1] > 0:
            pb = pb[:pb_peak[-1]+1]
        rp = np.max(pb) - np.min(pb)

        #find stop band
        sb = mag[f >= f0 * 1.2]
        sb_peak, _ = find_peaks(-sb)
        if sb_peak.size > 0 and sb_peak[0] < sb.size - 1:
            sb = sb[sb_peak[0]:]
        rs = np.min(pb) - np.max(sb)
    elif ftype == 'highpass':
        #find pass band
        pb = mag[f >= f0]
        pb_peak, _ = find_peaks(pb)
        if pb_peak.size > 0 and pb_peak[0] < pb.size - 1:
            pb = pb[pb_peak[0]:]
        rp = np.max(pb) - np.min(pb)

        #find stop band
        sb = mag[f <= f0 * 0.8]
        sb_peak, _ = find_peaks(-sb)
        if sb_peak.size > 0 and sb_peak[-1] > 0:
            sb = sb[:sb_peak[-1]+1]
        rs = np.min(pb) - np.max(sb)
    elif ftype == 'bandpass':
        #get band width
        bw = f0 / Q if Q else (f[-1] - f[0]) * 0.1

        #find pass band
        pb = mag[(f >= f0 - bw/2) & (f <= f0 + bw/2)]
        pb_peak, _ = find_peaks(pb)
        if pb_peak.size > 1:
            pb = pb[pb_peak[0]:pb_peak[-1]+1]
            mag_fc = np.min(pb)
        elif pb_peak.size == 1:
            mag_fc = pb[pb_peak[0]]
        else:
            mag_fc = np.max(pb)
        rp = np.max(pb) - mag_fc

        #find stop band
        sb1 = mag[(f < f0 - bw/2)]
        sb2 = mag[(f > f0 + bw/2)]
        sb1_peak, _ = find_peaks(-sb1)
        if sb1_peak.size > 0 and sb1_peak[-1] < sb1.size-1:
            sb1 = sb1[:sb1_peak[-1]+1]
        sb2_peak, _ = find_peaks(-sb2)
        if sb2_peak.size > 0 and sb2_peak[0] > 0:
            sb2 = sb2[sb1_peak[0]:]
        sb_max = max(np.max(sb1), np.max(sb2))
        rs = mag_fc - sb_max
    elif ftype == 'notch':
        #get band width
        bw = f0 / Q if Q else (f[-1] - f[0]) * 0.1

        #find pass band
        pb1 = mag[(f < f0 - bw/2)]
        pb2 = mag[(f > f0 + bw/2)]
        pb1_peak, _ = find_peaks(pb1)
        if pb1_peak.size > 0 and pb1_peak[-1] < pb1.size-1:
            pb1 = pb1[:pb1_peak[-1]+1]
        pb2_peak, _ = find_peaks(pb2)
        if pb2_peak.size > 0 and pb2_peak[0] > 0:
            pb2 = pb2[pb1_peak[0]:]
        pb_min = min(np.min(pb1), np.min(pb2))
        pb_max = max(np.max(pb1), np.max(pb2))
        rp = pb_max - pb_min

        #find stop band
        sb = mag[(f >= f0 - bw/2) & (f <= f0 + bw/2)]
        sb_peak, _ = find_peaks(-sb)
        if sb_peak.size > 1:
            sb = sb[sb_peak[0]:sb_peak[-1]+1]
            mag_fc = np.max(sb)
        elif sb_peak.size == 1:
            mag_fc = sb(sb_peak[0])
        else:
            mag_fc = np.min(sb)
        rs = pb_min - mag_fc
    else:
        rp = rs = None

    # 3. 群延遲平坦度
    w_gd, gd = group_delay((b, a), fs=fs)
    if ftype == 'lowpass':
        gd_pb = gd[w_gd <= f0]
    elif ftype == 'highpass':
        gd_pb = gd[w_gd >= f0]
    elif ftype in ['bandpass', 'notch']:
        bw = f0 / Q if Q else (f[-1] - f[0]) * 0.1
        mask = (w_gd >= f0 - bw/2) & (w_gd <= f0 + bw/2)
        gd_pb = gd[mask]
    else:
        gd_pb = gd
    gd_dev = gd_pb.max() - gd_pb.min()

    # 4. 極零分佈
    zeros, poles, _ = tf2zpk(b, a)

    # 5. 啟發式判斷設計方法
    method = 'unknown'
    if rp is not None:
        if rp < 0.1:
            method = 'bessel' if gd_dev < 1 else 'butterworth'
        elif rp >= 0.1 and rs is not None and rs < 1:
            method = 'cheby1'
        elif rs is not None and rs >= 1 and rp < 0.1:
            method = 'cheby2'
        elif rs is not None and rs >= 1 and rp >= 0.1:
            method = 'elliptic'

    return {
        'ftype':  ftype,
        'f0':     f0,
        'Q':      Q,
        'rp':     rp,
        'rs':     rs,
        'gd_dev': gd_dev,
        'poles':  poles,
        'zeros':  zeros,
        'method': method
    }
