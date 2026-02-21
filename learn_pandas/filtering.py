import pandas as pd

df = pd.read_csv("data.csv")

# Filtering = Keeping rows that meet a certain condition


df_fire = df[ df["Type 1"] == "Fire" ]  # Keep only rows where Type 1 is Fire

print("Fire type Pokemon:")
print(df_fire.head())


df_high_attack = df[ df["Attack"] > 100 ]  # Keep only rows where Attack is greater than 100
print("Pokemon with Attack greater than 100:")
print(df_high_attack.head().to_html())

df_legendary = df[ df["Legendary"] == 1 ]  # Keep only Legendary Pokemon
print("Legendary Pokemon:")
print(df_legendary.head().to_markdown())

df_nonwater_nonlegendary = df[( df['Type 1'] != "Water") | (df['Type 2'] != "Water") & (df["Legendary"] == 0)] 
# C-style operator for OR is | and for AND is &
print(f"Non-Water type Non-Legendary Pokemon: count={len(df_nonwater_nonlegendary)}")
print(df_nonwater_nonlegendary.head().to_string())