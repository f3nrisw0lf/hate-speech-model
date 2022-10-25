import re
from stop_words import remove_stop_words
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization  # to create AdamW optimizer

filipino_stopwords = [
    "akin", "aking", "ako", "alin",
    "am", "amin", "aming", "ang",
    "ano", "anumang", "apat", "at",
    "atin", "ating", "ay", "bababa",
    "bago", "bakit", "bawat", "bilang",
    "dahil", "dalawa", "dapat", "din",
    "dito", "doon", "gagawin", "gayunman",
    "ginagawa", "ginawa", "ginawang", "gumawa",
    "gusto", "habang", "hanggang", "hindi",
    "huwag", "iba", "ibaba", "ibabaw",
    "ibig", "ikaw", "ilagay", "ilalim",
    "ilan", "inyong", "isa", "isang",
    "itaas", "ito", "iyo", "iyon",
    "iyong", "ka", "kahit", "kailangan",
    "kailanman", "kami", "kanila", "kanilang",
    "kanino", "kanya", "kanyang", "kapag", "kapwa",
    "karamihan", "katiyakan", "katulad", "kaya",
    "kaysa", "ko", "kong", "kulang",
    "kumuha", "kung", "laban", "lahat",
    "lamang", "likod", "lima", "maaari",
    "maaaring", "maging", "mahusay", "makita",
    "marami", "marapat", "masyado", "may",
    "mayroon", "mga", "minsan", "mismo",
    "mula", "muli", "na", "nabanggit",
    "naging", "nagkaroon", "nais", "nakita",
    "namin", "napaka", "narito", "nasaan",
    "ng", "ngayon", "ni", "nila",
    "nilang", "nito", "niya", "niyang",
    "noon", "o", "pa", "paano",
    "pababa", "paggawa", "pagitan", "pagkakaroon",
    "pagkatapos", "palabas", "pamamagitan", "panahon",
    "pangalawa", "para", "paraan", "pareho",
    "pataas", "pero", "pumunta", "pumupunta",
    "sa", "saan", "sabi", "sabihin",
    "sarili", "sila", "sino", "siya",
    "tatlo", "tayo", "tulad", "tungkol",
    "una", "walang"
]

df = pd.read_csv('./hate_speech_cleansed/train_cleansed.csv', sep='|')
df_cleansed = df.dropna().reset_index(drop=True)

print(df_cleansed['text'].head(10))
df_cleansed.to_csv('dataset_v1.csv')
rnn_model = tf.keras.models.load_model('saved_model/rnn/v1', compile=False)

predictions = pd.DataFrame(rnn_model.predict(df_cleansed["text"]))
predictions.to_csv('predictions_v1.csv')
