import pandas as pd
import numpy as np


def envelope(signal, rate, threshold):
    mask = []
    y = pd.Series(signal).apply(np.abs)
    y_mean = y.rolling(window=int(rate/10), min_periods=1, center=True).mean()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)

    return mask


def calc_fft(signal, rate):
    n = len(signal)
    freq = np.fft.rfftfreq(n, d=1 / rate)
    Y = abs(np.fft.rfft(signal) / n)
    # print("======================================")
    # print(Y)
    # print("======================================")
    # print(freq)
    return Y, freq
