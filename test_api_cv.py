import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as tf_text

from official.nlp import optimization  # to create AdamW optimizer

tf.get_logger().setLevel('ERROR')
# tf.config.list_physical_devices('GPU')
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

df = pd.read_csv('./hate_speech_cleansed/train_cleansed.csv', sep='|')
df_cleansed = df.dropna().reset_index(drop=True)

# rnn_model = tf.keras.models.load_model('saved_model/rnn/v1', compile=False)

skf = StratifiedKFold(n_splits=10)

for index, value in enumerate(skf.split(df_cleansed['text'], df_cleansed['label'])):
    train_index, test_index = value
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = df_cleansed['text'][train_index], df_cleansed['text'][test_index]
    y_train, y_test = df_cleansed['label'][train_index], df_cleansed['label'][test_index]

    # Saving the Test Dataset for Performance Metrics
    text_df = pd.DataFrame(X_test)
    text_df = text_df.rename(columns={0: 'text'})

    label_df = pd.DataFrame(y_test)
    label_df = label_df.rename(columns={0: 'label'})

    df_testing_dataset = pd.concat([text_df, label_df], axis=1)
    # df_testing_dataset.to_csv(f'csv_rnn_test_dataset_fold_{index}.csv')

    # Testing
    # scores = rnn_model.predict(X_test, verbose=0)
    scores = requests.post('http://localhost:5000/many-hate-prediction',
                           json={"texts": list(X_test.to_numpy())})
    scores_array = list(map(lambda i: list(i.values())[0].get(
        'is_hate_speech'), scores.json()))
    # print(list(scores.json())[0].get('is_hate_speech'))
    score_df = pd.DataFrame(scores_array)
    score_df = score_df.rename(columns={0: 'prediction'})

    text_array = list(map(lambda i: list(i.values())[0].get(
        'original'), scores.json()))
    text_response_df = pd.DataFrame(text_array)
    text_response_df = text_response_df.rename(columns={0: 'original'})

    df_testing_dataset = pd.concat(
        [text_df, label_df, score_df, text_response_df], axis=1)

    df_testing_dataset.to_csv(f'csv_rnn_test_fold_{index}.csv')
    # np.savetxt(f"cv_rnn_score_fold_{index}", scores, delimiter=',')
