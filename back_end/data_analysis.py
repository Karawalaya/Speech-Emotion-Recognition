import numpy as np
import librosa
import back_end.calculations as calc
from python_speech_features import logfbank, mfcc
import back_end.plots_and_charts as pc
import matplotlib.pyplot as plt


def visual_analysis(df, envelope=True):
    df1 = df.copy()
    classes = list(np.unique(df1.emotion_label))

    signals = {}
    ffts = {}
    fbanks = {}
    mfccs = {}
    for c in classes:
        aud_fl_pth = df1[df1.emotion_label == c].iloc[0, 1]
        signal, rate = librosa.load(aud_fl_pth, sr=None)
        # print("==============")
        # print(signal)

        if envelope:
            mask = calc.envelope(signal, rate, 0.0005)
            signal = signal[mask]
        # print("@@@@@@@@@@@@@@")
        # print(signal)
        signals[c] = signal
        ffts[c] = calc.calc_fft(signal, rate)

        bank = logfbank(signal[:rate], rate, nfilt=26, nfft=1200).T
        fbanks[c] = bank

        mel = mfcc(signal[:rate], rate, numcep=13, nfilt=26, nfft=1200).T
        mfccs[c] = mel

    pc.plot_signals(signals)
    # plt.show()

    pc.plot_fft(ffts)
    # plt.show()

    pc.plot_fbank(fbanks)
    # plt.show()

    pc.plot_mfccs(mfccs)
    plt.show()

