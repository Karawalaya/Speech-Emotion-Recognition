import librosa
import os
import back_end.configurations as gconf
import back_end.calculations as calc
from python_speech_features import mfcc
import numpy as np


def predict(modelconfig, load_model, classes):
    aud_fl_list_recorded = os.listdir(gconf.uploads_dir)
    aud_fl_list_recorded.sort()
    file = aud_fl_list_recorded[-1]
    signal, rate = librosa.load(os.path.join(gconf.uploads_dir, file), sr=16000)

    mask = calc.envelope(signal, rate, 0.0005)
    signal = signal[mask]

    y_pred = []
    y_prob = []
    fn_prob = {}
    for i in range(0, signal.shape[0] - modelconfig.step, modelconfig.step):
        sample = signal[i:i + modelconfig.step]
        x = mfcc(sample, rate, numcep=modelconfig.nfeat, nfilt=modelconfig.nfilt, nfft=modelconfig.nfft)
        x = (x - modelconfig.min) / (modelconfig.max - modelconfig.min)

        x = x.reshape(1, x.shape[0], x.shape[1], 1)

        y_hat = load_model.predict(x)
        print(y_hat)
        print("ARGMAX:  ", np.argmax(y_hat))
        y_prob.append(y_hat)
        y_pred.append(np.argmax(y_hat))
    print("y_prob: ", y_prob)
    print("y_pred: ", y_pred)

    fn_prob[file] = np.mean(y_prob, axis=0).flatten()
    print("fn_prob[file]: ", fn_prob[file])

    y_probs = []
    y_prob = fn_prob[file]
    print("y_prob: ", y_prob)
    y_probs.append(y_prob)
    print("y_probs: ", y_probs)

    print(classes)
    y_pred = [classes[np.argmax(y)] for y in y_probs]
    print(y_pred)
    return y_pred
