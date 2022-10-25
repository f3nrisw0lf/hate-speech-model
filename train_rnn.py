from gc import callbacks
from stop_words import remove_stop_words
from tkinter import X
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as tf_text

from official.nlp import optimization  # to create AdamW optimizer

tf.get_logger().setLevel('ERROR')
tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

df = pd.read_csv('./hate_speech_cleansed/train_cleansed.csv', sep='|')
df_cleansed = df.dropna().reset_index(drop=True)
# remove_stop_words(df_cleansed['text'])

X_train, X_test, y_train, y_test = train_test_split(
    df_cleansed['text'], df_cleansed['label'], stratify=df_cleansed['label'])
# print(df_cleansed.head())

# Convert to TensorFlow Dataset
tf_dataset = tf.data.Dataset.from_tensor_slices(
    df_cleansed['text'].to_list())

tf_dataset = tf.data.Dataset.from_tensor_slices(X_train)

# Text Vectorization
encoder = tf.keras.layers.TextVectorization(
    max_tokens=None, output_mode='int', standardize="lower_and_strip_punctuation", split="whitespace")
encoder.adapt(tf_dataset)

vocab = np.array(encoder.get_vocabulary())


# Training the Model
model = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=64,
        # Use masking to handle the variable sequence lengths
        mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

early_stopper = tf.keras.callbacks.EarlyStopping(monitor="val_acc")
history = model.fit(X_train, y_train, epochs=50, callbacks=[early_stopper])

# Testing

# Saving the Test Dataset for Performance Metrics
text_df = pd.DataFrame(X_test)
text_df = text_df.rename(columns={0: 'text'})

label_df = pd.DataFrame(y_test)
label_df = label_df.rename(columns={0: 'label'})

df_testing_dataset = pd.concat([text_df, label_df], axis=1)

scores = model.predict(X_test, verbose=0)
score_df = pd.DataFrame(scores)
score_df = score_df.rename(columns={0: 'prediction'})
df_testing_dataset = pd.concat([text_df, label_df, score_df], axis=1)

df_testing_dataset.to_csv(f'csv_rnn_test.csv')

model.save('saved_model/rnn/v1')
# np.savetxt(f"cv_rnn_score_fold_{index}", scores, delimiter=',')
