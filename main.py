import pandas as pd
from preprocessing.data_processer import DataProcesser

df = pd.read_csv("datos/Resilience_CleanOnly_v1_PREPROCESSED.csv", encoding="latin1")

print(df.head())
