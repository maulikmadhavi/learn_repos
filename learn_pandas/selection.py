import pandas as pd
from fuzzywuzzy import process

df = pd.read_csv("data.csv") 

# SELECTION BY COLUMNS
# print(df["Name"].to_string())

# print(df["Attack"].to_string())

# SELECT MULTIPLE COLUMNS

# print(df[["Name","Attack"]].head().to_string())

# SELECTION BY ROWS

print(df.loc[3])  # By index id (generally if not specified loc and iloc behaves the same)
print(df.iloc[3]) # By integer location id

df = pd.read_csv("data.csv", index_col=["Name"]) #   You can set the index if you want
print(df.head())
SEL_COLS =  ["Type 1", "Type 2", "Attack","HP", "Defense", "Generation", "Speed"]
# Both are identical
print(df.loc["Bulbasaur"])
print(df.iloc[0])


# NEw try
print(df.loc["Pikachu"]) 
print(df.iloc[30])  # Found that Pikachu at line 31 (ignoring the 1st header row)

print(df.iloc[30][SEL_COLS])


# Checking the pokemon stats

x_search = input("Find the pokemon: ")




try:
    df_sel = df.loc[x_search][SEL_COLS]
    print(df_sel)
except KeyError:
    print(f"No matching found for {x_search}")
    best_match = process.extractOne(x_search, df.index.tolist(), score_cutoff=90)
    print(f"Best match is: {best_match}")
    # Do fuzzy match: 
    df_sel = df.loc[best_match[0]][SEL_COLS]
    print(df_sel)

    
