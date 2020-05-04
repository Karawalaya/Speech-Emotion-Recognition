import os
import back_end.data_loading as dl
import numpy as np


smpl_data_path_RAVDESS = '../../Resources/Datasets/RAVDESS_Audio/'
actr_dir_list_RAVDESS = os.listdir(smpl_data_path_RAVDESS)
clean_dir = 'clean'
tmp_test_dir = 'test'
uploads_dir = '../front_end/uploads'


class ModelConfig:
    def __init__(self, mode='convolutional', nfilt=26, nfeat=13, nfft=512, rate=16000):
        self.mode = mode
        self.nfilt = nfilt
        self.nfeat = nfeat
        self.nfft = nfft
        self.rate = rate
        self.step = int(rate / 10)
        self.model_path = os.path.join('models', mode + '.model')
        self.p_path = os.path.join('pickles', mode + '.p')


class RandFeatParams:
    def __init__(self, df):
        self.df1 = df.copy()
        self.df1 = dl.get_df_with_length(self.df1)
        self.n_samples = 2 * int(self.df1['length'].sum() / 0.1)
        self.class_dist = self.df1.groupby(['emotion_label'])['length'].mean()
        self.prob_dist = self.class_dist / self.class_dist.sum()
        self.classes = list(np.unique(self.df1.emotion_label))
