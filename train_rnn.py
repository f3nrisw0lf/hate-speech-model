import re
import string
from tkinter import X
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as tf_text

from official.nlp import optimization  # to create AdamW optimizer


def standardize(input_data):
    lowercase_str = tf.strings.lower(input_data)
    a_str = tf.strings.regex_replace(
        lowercase_str, f"[{re.escape(string.punctuation)}]", "")
    tokenizer = tf_text.WhitespaceTokenizer()
    tokens = tokenizer.tokenize(a_str)
    return tokens


def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history[metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, metric])


tf.get_logger().setLevel('ERROR')
# tf.config.list_physical_devices('GPU')
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

df = pd.read_csv('./hate_speech_cleansed/train_cleansed.csv', sep='|')
df_cleansed = df.dropna().reset_index(drop=True)

X_train, X_test, y_train, y_test = train_test_split(
    df_cleansed['text'], df_cleansed['label'], stratify=df_cleansed['label'])
# print(df_cleansed.head())

# Convert to TensorFlow Dataset
tf_dataset = tf.data.Dataset.from_tensor_slices(
    df_cleansed['text'].to_list())

# Text Vectorization
encoder = tf.keras.layers.TextVectorization(
    max_tokens=1000, output_mode='int', standardize="lower_and_strip_punctuation", split="whitespace")
encoder.adapt(tf_dataset)
vocab = np.array(encoder.get_vocabulary())
print(vocab[:20])

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

history = model.fit(X_train, y_train, epochs=1)
print(X_test)
scores = model.predict(X_test, verbose=0)
np.savetxt("score.csv", scores, delimiter=',')
# # Testing
# test_loss, test_acc = model.evaluate(X_test, y_test)
# print('Test Loss:', test_loss)
# print('Test Accuracy:', test_acc)
# print(history.history.keys())
# plt.figure(figsize=(16, 6))
# plt.subplot(1, 2, 1)
# plot_graphs(history, 'accuracy')
# plt.subplot(1, 2, 2)
# plot_graphs(history, 'loss')
# plt.show()
