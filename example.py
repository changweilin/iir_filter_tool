from iir_filter import design_iir, infer_iir_params, plot_response

params = {
        'ftype': 'bandpass',
        'f0': 1000,       # Hz，中心頻率
        'Q': 5,           # 僅 biquad 用
        'order': 2,
        'method': 'biquad',
        'rp': None,
        'rs': None
}
fs = 48000  # 取樣率

b, a = design_iir(params, fs)

# --- 列印設計與推論結果 ---
print("\n=== 設計輸入參數 ===")
for k, v in params.items():
    print(f"{k}: {v}")

print("\n=== 設計得到的濾波器係數 ===")
print("b:", b)
print("a:", a)

plot_response(b, a, fs=48000, title="原始設計")

# 推論回設計參數
inferred = infer_iir_params(b, a, fs)
b2, a2 = design_iir(inferred)

plot_response(b2, a2, fs=48000, title="反推重建")

print("\n=== 推論出的參數 ===")
for k, v in inferred.items():
    print(f"{k}: {v}")


print("\n=== 推論出的參數反推的濾波器係數 ===")
print("b2:", b2)
print("a2:", a2)