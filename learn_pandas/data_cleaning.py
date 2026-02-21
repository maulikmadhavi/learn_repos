import pandas as pd

# Data cleaning = the process of fixing/removing:
# - incorrect data
# - corrupted data
# - irrelevant data
# 70% of work in data science is cleaning the data


df = pd.read_csv("data.csv")

# 1. Drop irrelevant columns
IRRELEVANT_COLUMNS = ["#","Total","Sp. Atk","Generation","Sp. Def","Legendary"]
df = df.drop(columns=IRRELEVANT_COLUMNS)
print("Data after dropping irrelevant columns:")
print(df.head())

# 2. Handle missing values
# For example Charmader has missing "Type 2" value
df_fillna = df.fillna({"Type 2": "Unknown"}) # Fill missing "Type 2" values with "Unknown"
print("\nData after filling missing Type 2 values:")
print(df_fillna.head(10))
print(df_fillna["Type 2"].value_counts())

df_dropna = df.dropna(subset=["Type 2"]) # Drop the rows where "Type 2" is still null
print("\nData after dropping rows with missing Type 2 values:")
print(df_dropna.head(10))
print(df_dropna["Type 2"].value_counts())


# 3. Fix inconsistent values 
df["Type 1"] = df["Type 1"].replace({"Fire": "FIRE", "Water": "WATER"}) # Fix inconsistent "Type 1" values
print("\nData after fixing inconsistent Type 1 values:")
print(df.head(10))  
print(df["Type 1"].value_counts())


# 4. Standardize text data
df["Name"] = df["Name"].str.title() # Standardize "Name" column to title case
df["Type 1"] = df["Type 1"].str.capitalize() # Standardize "Type 1" column to capitalize case
df["Type 2"] = df["Type 2"].str.capitalize() # Standardize "Type 2" column to capitalize case
print("\nData after standardizing text data:")
print(df.head(10))


# 5. Fix data types
df["HP"] = df["HP"].astype(float) # Convert "HP" column to float type
print("\nData types after fixing HP column:")
print(df.dtypes)

# 6. Remove duplicates
df_no_duplicates = df.drop_duplicates() # Remove duplicate rows
print("\nData after removing duplicates:")
print(df_no_duplicates.head(10))