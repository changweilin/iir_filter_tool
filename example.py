from iir_filter import design_iir, infer_iir_params, plot_response


def print_params(title, params):
    print(f"\n=== {title} ===")
    for key, value in params.items():
        print(f"{key}: {value}")


def main():
    fs = 48000
    params = {
        "ftype": "bandpass",
        "f0": 1000,
        "Q": 5,
        "order": 2,
        "method": "biquad",
        "rp": None,
        "rs": None,
    }

    b, a = design_iir(params, fs)
    print_params("Input design parameters", params)
    print("\n=== Designed coefficients ===")
    print("b:", b)
    print("a:", a)
    plot_response(b, a, fs=fs, title="Original design")

    inferred = infer_iir_params(b, a, fs)
    print_params("Inferred best-effort parameters", inferred)

    if inferred.get("designable"):
        b2, a2 = design_iir(inferred, fs=fs)
        print("\n=== Redesigned coefficients from inferred parameters ===")
        print("b2:", b2)
        print("a2:", a2)
        plot_response(b2, a2, fs=fs, title="Best-effort redesign")
    else:
        print("\nInferred parameters are analysis-only and are not designable.")


if __name__ == "__main__":
    main()
