import json
import os
import re
import socket

import numpy as np
from flask import Flask, jsonify, render_template, request
from scipy.signal import freqz, sos2tf, tf2sos, tf2zpk, zpk2tf

from iir_filter import design_iir, infer_iir_params


DEFAULT_FS = 48000
RESPONSE_POINTS = 1024
MAX_RESPONSE_POINTS = 65536


def create_app():
    app = Flask(__name__)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.post("/api/design")
    def api_design():
        try:
            payload = request.get_json(silent=True) or {}
            params = _parse_design_payload(payload)
            fs = params["fs"]
            response_points = _response_points(payload)
            b, a = design_iir(params, fs=fs)
            return jsonify(
                {
                    "b": _json_safe(b),
                    "a": _json_safe(a),
                    "coefficients": _json_safe(_coefficient_representations(b, a)),
                    "response": _frequency_response(b, a, fs, response_points),
                }
            )
        except (TypeError, ValueError, KeyError) as exc:
            return jsonify({"error": str(exc)}), 400

    @app.post("/api/infer")
    def api_infer():
        try:
            payload = request.get_json(silent=True) or {}
            fs = _positive_float(payload.get("fs", DEFAULT_FS), "fs")
            response_points = _response_points(payload)
            b, a = _coefficients_from_payload(payload)
            inferred = infer_iir_params(b, a, fs)
            return jsonify(
                {
                    "inferred": _json_safe(inferred),
                    "coefficients": _json_safe(_coefficient_representations(b, a)),
                    "response": _frequency_response(b, a, fs, response_points),
                }
            )
        except (TypeError, ValueError, KeyError) as exc:
            return jsonify({"error": str(exc)}), 400

    return app


def _parse_design_payload(payload):
    return {
        "ftype": _required_string(payload, "ftype"),
        "method": _required_string(payload, "method"),
        "fs": _positive_float(payload.get("fs", DEFAULT_FS), "fs"),
        "f0": _positive_float(_required(payload, "f0"), "f0"),
        "Q": _optional_positive_float(payload.get("Q"), "Q"),
        "order": _positive_int(_required(payload, "order"), "order"),
        "rp": _optional_positive_float(payload.get("rp"), "rp"),
        "rs": _optional_positive_float(payload.get("rs"), "rs"),
    }


def _required(payload, key):
    if key not in payload or payload[key] in ("", None):
        raise ValueError(f"Missing required field: {key}")
    return payload[key]


def _required_string(payload, key):
    value = str(_required(payload, key)).strip()
    if not value:
        raise ValueError(f"Missing required field: {key}")
    return value


def _positive_float(value, name):
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not np.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{name} must be a positive finite value")
    return parsed


def _optional_positive_float(value, name):
    if value in ("", None):
        return None
    return _positive_float(value, name)


def _positive_int(value, name):
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def _response_points(payload):
    value = payload.get("response_points", RESPONSE_POINTS)
    points = _positive_int(value, "response_points")
    if points < 2 or points > MAX_RESPONSE_POINTS:
        raise ValueError(f"response_points must be between 2 and {MAX_RESPONSE_POINTS}")
    return points


def _parse_coefficients(value, name):
    if value in ("", None):
        raise ValueError(f"Missing required field: {name}")

    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(f"Missing required field: {name}")
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            value = [part for part in re.split(r"[\s,]+", text) if part]

    try:
        coefficients = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric coefficient array") from exc

    if coefficients.ndim != 1 or coefficients.size == 0:
        raise ValueError(f"{name} must be a one-dimensional coefficient array")
    if not np.all(np.isfinite(coefficients)):
        raise ValueError(f"{name} coefficients must be finite")
    return coefficients


def _coefficient_mode(payload):
    mode = str(payload.get("coefficient_mode", payload.get("mode", "tf"))).lower()
    if mode not in {"tf", "sos", "zpk"}:
        raise ValueError(f"Unsupported coefficient mode: {mode}")
    return mode


def _coefficients_from_payload(payload):
    mode = _coefficient_mode(payload)
    if mode == "tf":
        return _parse_coefficients(payload.get("b"), "b"), _parse_coefficients(payload.get("a"), "a")
    if mode == "sos":
        b, a = sos2tf(_parse_sos(payload.get("sos")))
        return _real_coefficients(b, "b"), _real_coefficients(a, "a")

    z, p, k = _parse_zpk(payload)
    b, a = zpk2tf(z, p, k)
    return _real_coefficients(b, "b"), _real_coefficients(a, "a")


def _parse_sos(value):
    if value is None or (isinstance(value, str) and value == ""):
        raise ValueError("Missing required field: sos")
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("Missing required field: sos")
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            value = [part for part in re.split(r"[\s,]+", text) if part]

    try:
        sos = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("sos must be a numeric second-order-section array") from exc

    if sos.ndim == 1 and sos.size % 6 == 0:
        sos = sos.reshape((-1, 6))
    if sos.ndim != 2 or sos.shape[1] != 6 or sos.shape[0] == 0:
        raise ValueError("sos must have shape (n_sections, 6)")
    if not np.all(np.isfinite(sos)):
        raise ValueError("sos coefficients must be finite")
    return sos


def _parse_zpk(payload):
    raw = payload.get("zpk")
    if raw is not None:
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError("zpk must be valid JSON") from exc
        if isinstance(raw, (list, tuple)) and len(raw) == 3:
            z, p, k = raw
        elif isinstance(raw, dict):
            z = raw.get("z", raw.get("zeros"))
            p = raw.get("p", raw.get("poles"))
            k = raw.get("k", raw.get("gain", 1))
        else:
            raise ValueError("zpk must be an object or [z, p, k] array")
    else:
        z = payload.get("z", payload.get("zeros"))
        p = payload.get("p", payload.get("poles"))
        k = payload.get("k", payload.get("gain", 1))

    zeros = _parse_complex_array(z, "z")
    poles = _parse_complex_array(p, "p")
    try:
        gain = complex(_parse_complex_value(k, "k"))
    except (TypeError, ValueError) as exc:
        raise ValueError("k must be a finite numeric gain") from exc
    if not np.isfinite(gain.real) or not np.isfinite(gain.imag):
        raise ValueError("k must be finite")
    return zeros, poles, gain


def _parse_complex_array(value, name):
    if value is None or (isinstance(value, str) and value == ""):
        return np.array([], dtype=complex)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return np.array([], dtype=complex)
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            value = [part for part in re.split(r"[\s,]+", text) if part]

    try:
        values = np.asarray([_parse_complex_value(item, name) for item in value], dtype=complex)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an array") from exc
    if values.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if not np.all(np.isfinite(values.real)) or not np.all(np.isfinite(values.imag)):
        raise ValueError(f"{name} values must be finite")
    return values


def _parse_complex_value(value, name):
    if isinstance(value, dict):
        real = value.get("real", value.get("re", 0))
        imag = value.get("imag", value.get("im", 0))
        return complex(float(real), float(imag))
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return complex(float(value[0]), float(value[1]))
    if isinstance(value, str):
        return complex(value.replace("i", "j").replace(" ", ""))
    try:
        return complex(float(value), 0.0)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} values must be finite numbers") from exc


def _real_coefficients(value, name):
    coefficients = np.real_if_close(np.asarray(value), tol=1000)
    try:
        coefficients = np.asarray(coefficients, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} coefficients must be real-valued") from exc
    if coefficients.ndim != 1 or coefficients.size == 0:
        raise ValueError(f"{name} must be a one-dimensional coefficient array")
    if not np.all(np.isfinite(coefficients)):
        raise ValueError(f"{name} coefficients must be finite")
    return coefficients


def _coefficient_representations(b, a):
    z, p, k = tf2zpk(b, a)
    return {
        "tf": {"b": b, "a": a},
        "sos": tf2sos(b, a),
        "zpk": {"z": z, "p": p, "k": k},
    }


def _frequency_response(b, a, fs, points=RESPONSE_POINTS):
    f, h = freqz(b, a, worN=points, fs=fs)
    magnitude = np.maximum(np.abs(h), np.finfo(float).tiny)
    magnitude_db = 20 * np.log10(magnitude)
    return {
        "frequency_hz": _json_safe(f),
        "magnitude_db": _json_safe(magnitude_db),
    }


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, complex):
        return {
            "real": _json_safe(value.real),
            "imag": _json_safe(value.imag),
        }
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _available_port(host, preferred_ports):
    for port in preferred_ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((host, port))
            except OSError:
                continue
            return port
    raise OSError(f"No available port found in {preferred_ports}")


def main():
    host = "127.0.0.1"
    if "PORT" in os.environ:
        ports = [int(os.environ["PORT"])]
    else:
        ports = [5000, 5001]
    port = _available_port(host, ports)
    debug = os.environ.get("FLASK_DEBUG") == "1"
    create_app().run(host=host, port=port, debug=debug, use_reloader=debug)


if __name__ == "__main__":
    main()
