import pandas as pd
import back_end.data_loading as dl
import librosa
import back_end.plots_and_charts as pc
import matplotlib.pyplot as plt
import back_end.classes_and_adjustments as ca
import back_end.data_analysis as da
import back_end.data_cleaning as dc
import back_end.configurations as conf
import back_end.build_features as bf
import numpy as np
import back_end.models as modl
from sklearn.utils.class_weight import compute_class_weight
from keras.callbacks import ModelCheckpoint
import back_end.verify_predict as vp
import os
import pickle
from keras.models import load_model
import back_end.predict as p


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

data_info_df = dl.load_data_intel(fromwhere='original')
# print(len(data_info_df))
# print(data_info_df.head())

# signal, rate = librosa.load(data_info_df.audio_file_path[500], sr=None)
# print(rate)
# print(data_info_df.audio_file_path[500])
# print(signal.ndim)
# print(signal.shape)
# print(signal.shape[0])
# print(signal.shape[0]/rate)
# print(signal)

# pc.plot_single_audio_wave(signal)
# pc.plot_single_audio_amplitude(signal, rate)
# pc.plot_single_audio_fft(signal, rate)
# plt.show()

# rate, signal = wavfile.read(data_info_df.audio_file_path[0])
# print(rate)
# rate, signal = wavfile.read(data_info_df.audio_file_path[400])
# print(rate)

df1 = ca.assign_emotion(data_info_df, 2)
# print(df1)
df1 = ca.isolate_by_gender(df1, gender='male')
# print(df1)
df1 = ca.remove_none_emotion(df1)
# print(df1)
df1 = ca.assign_classes(df1)
# print(len(df1))
# print(df1.head())

# pc.emotion_distribution_bar_plot(df1)
# pc.emotion_distribution_pie_plot(df1)     # check for the clean directory as well

# da.visual_analysis(df1, envelope=False)
# da.visual_analysis(df1, envelope=True)

dc.data_cleaning(df1)

mconf = conf.ModelConfig(mode='convolutional')
rfpconf = conf.RandFeatParams(df1)

X, y = bf.build_rand_feat(randfeatparams=rfpconf, modelconfig=mconf)
y_flat = np.argmax(y, axis=1)
input_shape = (X.shape[1], X.shape[2], 1)
model = modl.get_conv_model(input_shape)

class_weight = compute_class_weight('balanced', np.unique(y_flat), y_flat)

checkpoint = ModelCheckpoint(mconf.model_path, monitor='val_acc', mode='max', save_best_only=True, save_weights_only=False, period=1)

model.fit(X, y, epochs=60, batch_size=2048, shuffle=True, class_weight=class_weight, validation_split=0.1, callbacks=[checkpoint])

model.save(mconf.model_path)

vp.verification_predict(df1)

################################################################
p_path = os.path.join('pickles', 'convolutional.p')
with open(p_path, 'rb') as handle:
    modelconfig = pickle.load(handle)

loaded_model = load_model(modelconfig.model_path)
classes = list(np.unique(df1.emotion_label))
p.predict(modelconfig, loaded_model, classes)
