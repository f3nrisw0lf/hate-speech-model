import numpy as np
import pandas as pd
import csv

dataset = pd.read_csv('../hate_speech/valid.csv', sep=',', engine="python")
dataset_df = pd.DataFrame(dataset)

with open('valid_cleansed.csv', 'w', newline='') as file:
    writer = csv.writer(file, delimiter='|')
    writer.writerow(['text', 'label'])

    temp = ''
    for key, value in dataset_df.iterrows():
        if(not pd.isnull(value['label'])):
            writer.writerow([f"{temp} {value['text']}", value['label']])
            temp = ''
        else:
            temp = f"{temp} {value['text']}"

# print(
#     f"Value: {value['label']} Text: {value['text']} Type: {type(value['label'])}")
# print(f"Key: {key} Value: {value['text']}\n")
