from tqdm import tqdm
import os
import librosa
import numpy as np
from python_speech_features import mfcc
import back_end.configurations as gconf
import pickle
from keras.models import load_model
from sklearn.metrics import accuracy_score


def verification_predict(df):
    df1 = df.copy()
    classes = list(np.unique(df.emotion_label))
    fname_to_class = dict(zip(df.audio_fname, df.emotion_label))

    p_path = os.path.join('pickles', 'convolutional.p')
    with open(p_path, 'rb') as handle:
        modelconfig = pickle.load(handle)

    model = load_model(modelconfig.model_path)

    y_true, y_pred, fn_prob = build_predictions(classes=classes, fname_to_class=fname_to_class,
                                                modelconfig=modelconfig, model=model)
    acc_score = accuracy_score(y_true=y_true, y_pred=y_pred)

    y_probs = []
    for i, row in df1.iterrows():
        y_prob = fn_prob[row.audio_fname]
        y_probs.append(y_prob)
        for c, p in zip(classes, y_prob):
            df.at[i, c] = p

    y_pred = [classes[np.argmax(y)] for y in y_probs]
    df['y_pred'] = y_pred

    df.to_csv('predictions.csv', index=False)


def build_predictions(classes, fname_to_class, modelconfig, model):
    audio_dir = gconf.clean_dir

    y_true = []
    y_pred = []
    fn_prob = {}

    print('Extracting features from audio')
    for aud_fl in tqdm(os.listdir(audio_dir)):
        signal, rate = librosa.load(os.path.join(audio_dir, aud_fl), sr=None)
        emotion_label = fname_to_class[aud_fl]
        c =classes.index(emotion_label)
        y_prob = []

        for i in range(0, signal.shape[0]-modelconfig.step, modelconfig.step):
            sample = signal[i:i+modelconfig.step]
            x = mfcc(sample, rate, numcep=modelconfig.nfeat, nfilt=modelconfig.nfilt, nfft=modelconfig.nfft)
            x = (x - modelconfig.min) / (modelconfig.max - modelconfig.min)

            if modelconfig.mode == 'convolutional':
                x = x.reshape(1, x.shape[0], x.shape[1], 1)
            y_hat = model.predict(x)
            y_prob.append(y_hat)
            y_pred.append(np.argmax(y_hat))
            y_true.append(c)

        fn_prob[aud_fl] = np.mean(y_prob, axis=0).flatten()
        print("#####################@@@@@: ", fn_prob[aud_fl])

    return y_true, y_pred, fn_prob
