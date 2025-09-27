import pandas as pd
from procesador import DataProcesser

df = pd.read_csv("resultados/result.csv", encoding="latin1")

print(df.head())
