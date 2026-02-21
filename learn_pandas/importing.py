import pandas as pd


# df = pd.read_csv("data.csv")
df = pd.read_json("data.json")

df = df[["id","name","type","species"]]
print(df.head().to_string())

print(df.head())