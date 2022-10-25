import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
from IPython.display import clear_output

import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization  # to create AdamW optimizer


class PlotLearning(keras.callbacks.Callback):
    """
    Callback to plot the learning curves of the model during training.
    """

    def on_train_begin(self, logs={}):
        self.metrics = {}
        for metric in logs:
            self.metrics[metric] = []

    def on_epoch_end(self, epoch, logs={}):
        # Storing metrics
        for metric in logs:
            if metric in self.metrics:
                self.metrics[metric].append(logs.get(metric))
            else:
                self.metrics[metric] = [logs.get(metric)]

        # Plotting
        metrics = [x for x in logs if 'val' not in x]

        f, axs = plt.subplots(1, len(metrics), figsize=(15, 5))
        clear_output(wait=True)

        for i, metric in enumerate(metrics):
            axs[i].plot(range(1, epoch + 2),
                        self.metrics[metric],
                        label=metric)
            if logs['val_' + metric]:
                axs[i].plot(range(1, epoch + 2),
                            self.metrics['val_' + metric],
                            label='val_' + metric)

            axs[i].legend()
            axs[i].grid()

        plt.tight_layout()
        plt.show()


def train_bert(X_train, y_train, callbacks):
    # Preprocess model auto-selected: https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3
    bert_preprocess = hub.KerasLayer(
        "bert_pre_req\\bert_en_uncased_preprocess_3\\")

    # BERT model selected           : https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1
    bert_encoder = hub.KerasLayer(
        "bert_pre_req\\bert_en_uncased_L-12_H-768_A-12_4\\")

    # Bert layers
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessed_text = bert_preprocess(text_input)
    outputs = bert_encoder(preprocessed_text)

    # Neural network layers
    l = tf.keras.layers.Dropout(0.1, name="dropout")(outputs['pooled_output'])
    l = tf.keras.layers.Dense(1, activation='sigmoid', name="output")(l)

    # Use inputs and outputs to construct a final model
    model = tf.keras.Model(inputs=[text_input], outputs=[l])

    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=5,
              batch_size=32, verbose=1, validation_data=(X_test, y_test), callbacks=callbacks)

    # model.save("bert")
    model.summary()

    return model


tf.get_logger().setLevel('ERROR')

tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

df = pd.read_csv('./hate_speech_cleansed/train_cleansed.csv',
                 sep='|', encoding='latin')

df_cleansed = df.dropna().reset_index(drop=True)

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

    # df_testing_dataset = pd.concat([text_df, label_df], axis=1)
    # df_testing_dataset.to_csv(f'csv_bert_test_dataset_fold_{index}.csv')

    early_stopper = tf.keras.callbacks.EarlyStopping(monitor="val_acc")
    print(f"X-Train: {X_train} | X-Test: {X_test}")

    model = train_bert(X_train=X_train, y_train=y_train,
                       callbacks=[early_stopper])

    # Testing
    scores = model.predict(X_test, verbose=0)
    score_df = pd.DataFrame(scores)
    score_df = score_df.rename(columns={0: 'prediction'})
    df_testing_dataset = pd.concat([text_df, label_df, score_df], axis=1)

    df_testing_dataset.to_csv(f'csv_bert_test_fold_{index}.csv')
    # np.savetxt(f"cv_bert_score_fold_{index}", scores, delimiter=',')
