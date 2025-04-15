import matplotlib.pyplot as plt
from scipy.signal import freqz
import numpy as np

def plot_response(b, a, fs, title=None):
    f, h = freqz(b, a, fs=fs)
    mag = 20 * np.log10(np.abs(h))
    plt.figure()
    plt.plot(f, mag)
    plt.title(title or "頻率響應")
    plt.xlabel("頻率 (Hz)")
    plt.ylabel("幅度 (dB)")
    plt.grid()
    plt.show()