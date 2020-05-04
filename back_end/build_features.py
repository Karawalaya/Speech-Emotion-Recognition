import os
import pickle
from tqdm import tqdm
import numpy as np
import librosa
from python_speech_features import mfcc
import back_end.configurations as gconf
from keras.utils import to_categorical


def check_data(modelconfig):

    if os.path.isfile(modelconfig.p_path):
        print('Loading existing data for {} model'.format(modelconfig.mode))
        with open(modelconfig.p_path, 'rb') as infile:
            tmp = pickle.load(infile)
            return tmp
    else:
        return None


def build_rand_feat(randfeatparams, modelconfig):
    tmp = check_data(modelconfig)
    if tmp:
        return tmp.features[0], tmp.features[1]

    df = randfeatparams.df1
    n_samples = randfeatparams.n_samples
    print("---------------------", n_samples)
    class_dist = randfeatparams.class_dist
    prob_dist = randfeatparams.prob_dist
    classes = randfeatparams.classes
    a = range(n_samples)[-1]
    X = []
    y = []
    _min, _max = float('inf'), -float('inf')
    for _ in tqdm(range(n_samples)):
        rand_class = np.random.choice(class_dist.index, p=prob_dist)
        file = np.random.choice(df[df.emotion_label == rand_class].audio_fname)
        # print(file)
        signal, rate = librosa.load(gconf.clean_dir + '/' + file, sr=None)
        if _ == a:
            print("signal---------------------", signal.shape)
        # print(rand_class)
        # print(df.loc[df['audio_fname'] == file, 'emotion_label'].iloc[0])
        emotion_label = df.loc[df['audio_fname'] == file, 'emotion_label'].iloc[0]
        rand_index = np.random.randint(0, signal.shape[0] - modelconfig.step)
        sample = signal[rand_index:rand_index+modelconfig.step]
        if _ == a:
            print("sample---------------------", sample.shape)
        X_sample = mfcc(sample, rate, numcep=modelconfig.nfeat, nfilt=modelconfig.nfilt, nfft=modelconfig.nfft)
        if _ == a:
            print("X_sample---------------------", X_sample.shape)
        _min = min(np.amin(X_sample), _min)
        _max = max(np.amax(X_sample), _max)
        X.append(X_sample)
        if _ == a:
            print("X---------------------", len(X))
        y.append(classes.index(emotion_label))
    modelconfig.min = _min
    modelconfig.max = _max
    X, y = np.array(X), np.array(y)
    print("@@@@@@@    ", X.shape)
    X = (X - _min) / (_max - _min)
    print("#########   ", X.shape)
    print("#########   ", y.shape)
    if modelconfig.mode == 'convolutional':
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)    #  X [no_samples, 13 ,9 ,1]
        print("^^^^^^^^^^      ", X.shape)
        print("^^^^^^^^^^ Should be similar to no of samples     ", X.shape[0])

    y = to_categorical(y, num_classes=len(classes))

    modelconfig.features = (X, y)   # python allows you to add new fields to the objects on the fly.

    with open(modelconfig.p_path, 'wb') as outfile:
        pickle.dump(modelconfig, outfile, protocol=2)

    return X, y