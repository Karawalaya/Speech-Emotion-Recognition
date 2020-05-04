def assign_emotion(df, class_num):
    df1 = df.copy()
    # print(df)
    if class_num <= 2 or (3 < class_num < 8) or class_num > 8:
        df2 = emotion_class_2(df1)
    elif class_num == 3:
        df2 = emotion_class_3(df1)
    elif class_num == 8:
        df2 = emotion_class_8(df1)

    return df2


def emotion_class_2(df):
    all_emotions_list = []
    for index in range(len(df)):
        if df.emotion_no[index] == 2:
            act_emotion = "positive"
        elif df.emotion_no[index] == 3:
            act_emotion = "positive"
        elif df.emotion_no[index] == 4:
            act_emotion = "negative"
        elif df.emotion_no[index] == 5:
            act_emotion = "negative"
        elif df.emotion_no[index] == 6:
            act_emotion = "negative"
        else:
            act_emotion = "none"

        all_emotions_list.append(act_emotion)
    df['emotion'] = all_emotions_list
    return df


def emotion_class_3(df):
    all_emotions_list = []
    for index in range(len(df)):
        if df.emotion_no[index] == 1:
            act_emotion = "neutral"
        elif df.emotion_no[index] == 2:
            act_emotion = "neutral"
        elif df.emotion_no[index] == 3:
            act_emotion = "positive"
        elif df.emotion_no[index] == 4:
            act_emotion = "negative"
        elif df.emotion_no[index] == 5:
            act_emotion = "negative"
        elif df.emotion_no[index] == 6:
            act_emotion = "negative"
        else:
            act_emotion = "none"

        all_emotions_list.append(act_emotion)
    df['emotion'] = all_emotions_list
    return df


def emotion_class_8(df):
    all_emotions_list = []
    for index in range(len(df)):
        if df.emotion_no[index] == 1:
            act_emotion = "neutral"
        elif df.emotion_no[index] == 2:
            act_emotion = "calm"
        elif df.emotion_no[index] == 3:
            act_emotion = "happy"
        elif df.emotion_no[index] == 4:
            act_emotion = "sad"
        elif df.emotion_no[index] == 5:
            act_emotion = "angry"
        elif df.emotion_no[index] == 6:
            act_emotion = "fearful"
        elif df.emotion_no[index] == 7:
            act_emotion = "disgust"
        elif df.emotion_no[index] == 8:
            act_emotion = "surprised"
        else:
            act_emotion = "none"

        all_emotions_list.append(act_emotion)
    df['emotion'] = all_emotions_list
    return df


def isolate_by_gender(df, gender):
    df1 = df.copy()
    if gender == 'female':
        df2 = isolate_by_female(df1)
    elif gender == 'male':
        df2 = isolate_by_male(df1)

    return df2


def isolate_by_female(df):
    df1 = df.copy()
    df1 = df1[df1.gender != "male"]
    df1.reset_index(drop=True, inplace=True)

    return df1


def isolate_by_male(df):
    df1 = df.copy()
    df1 = df1[df1.gender != "female"]
    df1.reset_index(drop=True, inplace=True)

    return df1


def remove_none_emotion(df):
    df1 = df.copy()
    df1 = df1[df1.emotion != "none"]
    df1.reset_index(drop=True, inplace=True)

    return df1


def assign_classes(df):
    df1 = df.copy()
    all_emotions_label_list = []
    for index in range(len(df1)):
        all_emotions_label_list.append(df1.gender[index] + '_' + df1.emotion[index])
    df1['emotion_label'] = all_emotions_label_list

    return df1
