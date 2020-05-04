import numpy as np
import os
import back_end.configurations as gconf
from tqdm import tqdm
import librosa
import back_end.calculations as clac
from scipy.io import wavfile


def data_cleaning(df):
    df1 = df.copy()
    classes = list(np.unique(df1.emotion_label))

    # clean_dir = os.path.join(os.getcwd(), 'clean1')
    if not os.path.isdir(gconf.clean_dir):
        os.makedirs(gconf.clean_dir)

    if len(os.listdir(gconf.clean_dir)) == 0:
        for aud_fl_pth in tqdm(df1.audio_file_path):
            # print(df1.audio_fname[df1.audio_file_path == aud_fl_pth])
            signal, rate = librosa.load(aud_fl_pth, sr=16000)
            mask = clac.envelope(signal, rate, 0.0005)
            # print('clean/' + df1[df1.audio_file_path == aud_fl_pth].iloc[0, 0])
            wavfile.write(filename=gconf.clean_dir + '/' + df1[df1.audio_file_path == aud_fl_pth].iloc[0, 0], rate=rate, data=signal[mask])
