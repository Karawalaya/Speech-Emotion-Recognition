import pandas as pd
import os
import back_end.configurations as gconf
from tqdm import tqdm
import librosa


def load_data_intel(fromwhere='original'):
    data_info_df = pd.DataFrame(columns=['audio_fname', 'audio_file_path', 'actor_no', 'gender', 'emotion_no'])
    count = 0

    if fromwhere == 'original':
        for act_dir in gconf.actr_dir_list_RAVDESS:
            aud_fl_list_RAVDESS = os.listdir(gconf.smpl_data_path_RAVDESS + act_dir)
            for aud_fl in aud_fl_list_RAVDESS:
                ind_aud_fl_ids = aud_fl.split('.')[0].split('-')
                aud_fl_pth = gconf.smpl_data_path_RAVDESS + act_dir + '/' + aud_fl
                fname = aud_fl
                actor_no = int(ind_aud_fl_ids[-1])
                emotion_no = int(ind_aud_fl_ids[2])

                if int(actor_no) % 2 == 0:
                    gender = "female"
                else:
                    gender = "male"

                data_info_df.loc[count] = [fname, aud_fl_pth, actor_no, gender, emotion_no]
                count += 1
        print('Data successfully loaded from the original directory')

    elif fromwhere == 'clean':
        aud_fl_list_RAVDESS = os.listdir(gconf.clean_dir + '/')
        for aud_fl in tqdm(aud_fl_list_RAVDESS):
            ind_aud_fl_ids = aud_fl.split('.')[0].split('-')
            aud_fl_pth = gconf.clean_dir + '/' + aud_fl
            fname = aud_fl
            actor_no = int(ind_aud_fl_ids[-1])
            emotion_no = int(ind_aud_fl_ids[2])

            if int(actor_no) % 2 == 0:
                gender = "female"
            else:
                gender = "male"

            data_info_df.loc[count] = [fname, aud_fl_pth, actor_no, gender, emotion_no]
            count += 1
        print('Data successfully loaded from the "clean" directory')

    return data_info_df


def get_df_with_length(df):
    df1 = df.copy()
    '''
        for index in range(len(data_info_df)):
            rate, signal = wavfile.read(data_info_df.audio_file_path[index])
            data_info_df.


        '''

    df1.set_index('audio_file_path', inplace=True)

    # rate, signal = wavfile.read(data_info_df.audio_file_path[400]) # 0, 1, 500
    # print(rate)
    # rate, signal = wavfile.read(data_info_df.audio_file_path[401])
    # print(rate)

    for audio_file_path in df1.index:
        signal, rate = librosa.load(audio_file_path, sr=None)
        # print(rate)
        df1.at[audio_file_path, 'length'] = signal.shape[0] / rate

    df1.reset_index(inplace=True)
    return df1