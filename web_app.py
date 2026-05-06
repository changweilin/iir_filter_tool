import json
import os
import re
import socket

import numpy as np
from flask import Flask, jsonify, render_template, request
from scipy.signal import freqz

from iir_filter import design_iir, infer_iir_params


DEFAULT_FS = 48000
RESPONSE_POINTS = 1024


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
            b, a = design_iir(params, fs=fs)
            inferred = infer_iir_params(b, a, fs)
            return jsonify(
                {
                    "b": _json_safe(b),
                    "a": _json_safe(a),
                    "response": _frequency_response(b, a, fs),
                    "inferred": _json_safe(inferred),
                }
            )
        except (TypeError, ValueError, KeyError) as exc:
            return jsonify({"error": str(exc)}), 400

    @app.post("/api/infer")
    def api_infer():
        try:
            payload = request.get_json(silent=True) or {}
            fs = _positive_float(payload.get("fs", DEFAULT_FS), "fs")
            b = _parse_coefficients(payload.get("b"), "b")
            a = _parse_coefficients(payload.get("a"), "a")
            inferred = infer_iir_params(b, a, fs)
            return jsonify(
                {
                    "inferred": _json_safe(inferred),
                    "response": _frequency_response(b, a, fs),
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


def _frequency_response(b, a, fs):
    f, h = freqz(b, a, worN=RESPONSE_POINTS, fs=fs)
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
