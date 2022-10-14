import numpy as np
import pandas as pd

dataset = pd.read_csv('../hate_speech/valid.csv', sep=',', engine="python")
dataset_df = pd.DataFrame(dataset)

print(dataset_df.shape)

for key, value in dataset_df.head(100).iterrows():
    print(f"Key: {key} Label: {value['label']} Text: {value['text']}")
