import numpy as np
from scipy.signal import bessel, butter, cheby1, cheby2, ellip


_METHOD_ALIASES = {
    "biquad": "biquad",
    "butter": "butterworth",
    "butterworth": "butterworth",
    "cheby1": "cheby1",
    "cheby2": "cheby2",
    "ellip": "elliptic",
    "elliptic": "elliptic",
    "bessel": "bessel",
}

_FTYPE_ALIASES = {
    "lowpass": "lowpass",
    "highpass": "highpass",
    "bandpass": "bandpass",
    "notch": "notch",
    "bandstop": "notch",
}


def _require_key(info, key):
    if key not in info:
        raise ValueError(f"Missing required filter parameter: {key}")
    return info[key]


def _as_positive_float(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite value")
    return value


def _normalize_ftype(ftype):
    normalized = _FTYPE_ALIASES.get(str(ftype).lower())
    if normalized is None:
        raise ValueError(f"Unsupported filter type: {ftype}")
    return normalized


def _normalize_method(method):
    normalized = _METHOD_ALIASES.get(str(method).lower())
    if normalized is None:
        raise ValueError(f"Unsupported design method: {method}")
    return normalized


def _band_edges(f0, q, nyq):
    bandwidth = f0 / q if q is not None else f0 * 0.1
    low = max((f0 - bandwidth / 2.0) / nyq, 1e-6)
    high = min((f0 + bandwidth / 2.0) / nyq, 0.999)
    if low >= high:
        raise ValueError("Computed band edges are invalid; check f0, Q, and fs")
    return [low, high]


def design_iir(info, fs=None):
    """
    Design an IIR filter from a small parameter dictionary.

    Supported public filter types are lowpass, highpass, bandpass, and notch.
    The SciPy-backed design path maps notch to SciPy's bandstop btype.
    """
    order = int(_require_key(info, "order"))
    ftype = _normalize_ftype(_require_key(info, "ftype"))
    f0 = _as_positive_float(_require_key(info, "f0"), "f0")
    q = info.get("Q")
    q = _as_positive_float(q, "Q") if q is not None else None
    method = _normalize_method(_require_key(info, "method"))
    rp = info.get("rp")
    rs = info.get("rs")

    fs = _as_positive_float(fs if fs is not None else info.get("fs", 48000), "fs")
    nyq = fs / 2.0
    if f0 >= nyq:
        raise ValueError("f0 must be below the Nyquist frequency")

    if method == "biquad":
        if order != 2:
            raise ValueError("RBJ biquad design requires order=2")
        if q is None:
            raise ValueError("Q is required for RBJ biquad design")

        w0 = 2 * np.pi * f0 / fs
        alpha = np.sin(w0) / (2 * q)
        cosp = np.cos(w0)
        if ftype == "lowpass":
            b0, b1, b2 = (1 - cosp) / 2, 1 - cosp, (1 - cosp) / 2
        elif ftype == "highpass":
            b0, b1, b2 = (1 + cosp) / 2, -(1 + cosp), (1 + cosp) / 2
        elif ftype == "bandpass":
            b0, b1, b2 = alpha, 0, -alpha
        elif ftype == "notch":
            b0, b1, b2 = 1, -2 * cosp, 1
        else:
            raise ValueError(f"Unsupported biquad filter type: {ftype}")

        a0 = 1 + alpha
        a1 = -2 * cosp
        a2 = 1 - alpha
        b = np.array([b0, b1, b2]) / a0
        a = np.array([1, a1 / a0, a2 / a0])
        return b, a

    if ftype in ("lowpass", "highpass"):
        wn = f0 / nyq
        scipy_btype = ftype
    else:
        wn = _band_edges(f0, q, nyq)
        scipy_btype = "bandstop" if ftype == "notch" else "bandpass"

    if method == "butterworth":
        return butter(order, wn, btype=scipy_btype, analog=False)
    if method == "cheby1":
        if rp is None:
            raise ValueError("rp is required for cheby1 design")
        return cheby1(order, rp, wn, btype=scipy_btype, analog=False)
    if method == "cheby2":
        if rs is None:
            raise ValueError("rs is required for cheby2 design")
        return cheby2(order, rs, wn, btype=scipy_btype, analog=False)
    if method == "elliptic":
        if rp is None or rs is None:
            raise ValueError("rp and rs are required for elliptic design")
        return ellip(order, rp, rs, wn, btype=scipy_btype, analog=False)
    if method == "bessel":
        return bessel(order, wn, btype=scipy_btype, analog=False)

    raise ValueError(f"Unsupported design method: {method}")
