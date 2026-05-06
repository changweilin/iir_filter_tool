import warnings

import numpy as np
from scipy.signal import freqz, group_delay, sos2tf, tf2sos, tf2zpk


def convert_to_biquads(b, a):
    """Convert transfer-function coefficients to second-order sections."""
    return tf2sos(b, a)


def convert_from_biquads(sos):
    """Convert second-order sections to transfer-function coefficients."""
    return sos2tf(sos)


def _db_magnitude(response):
    magnitude = np.maximum(np.abs(response), np.finfo(float).tiny)
    return 20 * np.log10(magnitude)


def _finite_values(values):
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def _safe_min(values):
    values = _finite_values(values)
    return float(np.min(values)) if values.size else None


def _safe_max(values):
    values = _finite_values(values)
    return float(np.max(values)) if values.size else None


def _safe_range(values):
    values = _finite_values(values)
    return float(np.max(values) - np.min(values)) if values.size else None


def _attenuation(reference_values, rejected_values):
    reference_min = _safe_min(reference_values)
    rejected_max = _safe_max(rejected_values)
    if reference_min is None or rejected_max is None:
        return None
    return float(reference_min - rejected_max)


def _bandwidth_from_q(f, f0, q):
    if f0 is None:
        return None
    if q is not None and q > 0:
        return f0 / q
    if f.size < 2:
        return None
    span = f[-1] - f[0]
    return span * 0.1 if span > 0 else None


def _estimate_q(f0, f1, f2):
    if f0 is None or f1 is None or f2 is None or f2 <= f1:
        return None
    return float(f0 / (f2 - f1))


def analyze_iir(b, a, fs, worN=8000):
    """
    Analyze a digital IIR response and estimate ftype, f0, and Q.

    The estimates are intentionally best-effort. Missing threshold crossings
    produce partial metadata instead of raising.
    """
    f, h = freqz(b, a, worN=worN, fs=fs)
    mag = _db_magnitude(h)
    if f.size == 0:
        return f, mag, "unknown", None, None

    mag0 = mag[0]
    mag_nyquist = mag[-1]
    idx_peak = int(np.argmax(mag))
    idx_valley = int(np.argmin(mag))
    delta = mag0 - mag_nyquist

    if (
        0 < idx_peak < f.size - 1
        and mag[idx_peak] - max(mag0, mag_nyquist) > 3
    ):
        f0 = float(f[idx_peak])
        threshold = mag[idx_peak] - 3
        left = np.flatnonzero(mag[:idx_peak] <= threshold)
        right = np.flatnonzero(mag[idx_peak:] <= threshold)
        f1 = float(f[left[-1]]) if left.size else None
        f2 = float(f[idx_peak + right[0]]) if right.size else None
        return f, mag, "bandpass", f0, _estimate_q(f0, f1, f2)

    if (
        0 < idx_valley < f.size - 1
        and min(mag0, mag_nyquist) - mag[idx_valley] > 3
    ):
        f0 = float(f[idx_valley])
        threshold = mag[idx_valley] + 3
        left = np.flatnonzero(mag[:idx_valley] >= threshold)
        right = np.flatnonzero(mag[idx_valley:] >= threshold)
        f1 = float(f[left[-1]]) if left.size else None
        f2 = float(f[idx_valley + right[0]]) if right.size else None
        return f, mag, "notch", f0, _estimate_q(f0, f1, f2)

    if delta > 3:
        threshold = mag0 - 3
        crossings = np.flatnonzero(mag <= threshold)
        f0 = float(f[crossings[0]]) if crossings.size else None
        return f, mag, "lowpass", f0, None

    if delta < -3:
        threshold = mag_nyquist - 3
        crossings = np.flatnonzero(mag >= threshold)
        f0 = float(f[crossings[0]]) if crossings.size else None
        return f, mag, "highpass", f0, None

    return f, mag, "unknown", None, None


def _estimate_ripple_and_stopband(f, mag, ftype, f0, q):
    if f0 is None:
        return None, None

    if ftype == "lowpass":
        passband = mag[f <= f0]
        stopband = mag[f >= f0 * 1.2]
        return _safe_range(passband), _attenuation(passband, stopband)

    if ftype == "highpass":
        passband = mag[f >= f0]
        stopband = mag[f <= f0 * 0.8]
        return _safe_range(passband), _attenuation(passband, stopband)

    bandwidth = _bandwidth_from_q(f, f0, q)
    if bandwidth is None:
        return None, None

    low = f0 - bandwidth / 2
    high = f0 + bandwidth / 2
    band_mask = (f >= low) & (f <= high)

    if ftype == "bandpass":
        passband = mag[band_mask]
        stopband = mag[~band_mask]
        return _safe_range(passband), _attenuation(passband, stopband)

    if ftype == "notch":
        passband = mag[~band_mask]
        stopband = mag[band_mask]
        return _safe_range(passband), _attenuation(passband, stopband)

    return None, None


def _estimate_group_delay_deviation(b, a, fs, f, ftype, f0, q):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            w_gd, gd = group_delay((b, a), fs=fs)
        except Exception:
            return None

    if ftype == "lowpass" and f0 is not None:
        gd_band = gd[w_gd <= f0]
    elif ftype == "highpass" and f0 is not None:
        gd_band = gd[w_gd >= f0]
    elif ftype in ("bandpass", "notch") and f0 is not None:
        bandwidth = _bandwidth_from_q(f, f0, q)
        if bandwidth is None:
            gd_band = gd
        else:
            low = f0 - bandwidth / 2
            high = f0 + bandwidth / 2
            gd_band = gd[(w_gd >= low) & (w_gd <= high)]
    else:
        gd_band = gd

    return _safe_range(gd_band)


def _infer_method(rp, rs, gd_dev):
    if rp is None or gd_dev is None:
        return "unknown"
    if rs is not None and (not np.isfinite(rs) or rs > 300):
        return "unknown"
    if rp < 0.1:
        return "bessel" if gd_dev < 1 else "butterworth"
    if rs is not None and rs < 1:
        return "cheby1"
    if rs is not None and rp < 0.1:
        return "cheby2"
    if rs is not None:
        return "elliptic"
    return "unknown"


def infer_iir_params(b, a, fs):
    """
    Infer best-effort descriptive parameters from IIR coefficients.

    The return value is analysis metadata. It may be designable, but exact
    round-trip reconstruction is not guaranteed.
    """
    b = np.asarray(b, dtype=float)
    a = np.asarray(a, dtype=float)
    fs = float(fs)

    f, mag, ftype, f0, q = analyze_iir(b, a, fs)
    rp, rs = _estimate_ripple_and_stopband(f, mag, ftype, f0, q)
    gd_dev = _estimate_group_delay_deviation(b, a, fs, f, ftype, f0, q)

    try:
        zeros, poles, _ = tf2zpk(b, a)
    except Exception:
        zeros = np.array([])
        poles = np.array([])

    method = _infer_method(rp, rs, gd_dev)
    order = max(len(a) - 1, 0)
    designable = ftype != "unknown" and f0 is not None and method != "unknown" and order > 0

    return {
        "ftype": ftype,
        "f0": f0,
        "Q": q,
        "order": order,
        "fs": fs,
        "rp": rp,
        "rs": rs,
        "gd_dev": gd_dev,
        "poles": poles,
        "zeros": zeros,
        "method": method,
        "designable": designable,
    }
