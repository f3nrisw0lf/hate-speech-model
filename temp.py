import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization  # to create AdamW optimizer

df = pd.read_csv('./hate_speech/train.csv', sep=',')
df_cleansed = df.dropna().reset_index(drop=True)
loaded_model = tf.keras.models.load_model('./bert')

loaded_model.summary()
prediction = loaded_model.predict(
    ["ISA LANG MASASABI KO SA AD NI BINAY, P U T A. Bobo lang ang boboto kay Binay.",
     "I am not for Binay. I am sharing soundbites from what we've gathered in going around barangays.",
     "Nakakasuka! Blergh!! #LeniOnMMK #MMKIna #RoxasRobredo2016 #ABSCBNFAIL"])

print(f"Prediction: {prediction}")
