import pandas as pd

# Aggregation = Performing a calculation on a set of values to return a single value
# Usually it is done with groupby() function

df = pd.read_csv("data.csv")


# # ====== Whole DataFrame stats ======
# average_df = df.mean(numeric_only=True)
# print("Average stats of all Pokemon:")
# print(average_df)


# sum_df = df.sum(numeric_only=True)
# print("\nSum stats of all Pokemon:")
# print(sum_df)

# min_df = df.min(numeric_only=True)
# print("\nMinimum stats of all Pokemon:")
# print(min_df)

# max_df = df.max(numeric_only=True)
# print("\nMaximum stats of all Pokemon:")
# print(max_df)

# count_df = df.count()
# print("\nCount of non-null entries in each column:")
# print(count_df) # Few entries may be less if there are null values

# # ======  single column ======
# df_attack = df["Attack"].mean()
# print(f"\nAverage Attack of all Pokemon: {df_attack}")  
# df_hp = df["HP"].max()
# print(f"Maximum HP of all Pokemon: {df_hp}")    

# ======== Group by Type 1 and get average stats ======
# print(df[["Type 1","Speed"]].head().to_string())

grouped_type1 = df.groupby("Type 1")
# grouped_type1 = df.groupby("Type 1").mean(numeric_only=True)
print("\nAverage stats grouped by Type 1:")
print(grouped_type1.head())
print(grouped_type1.size())
# print(grouped_type1["Legendary"].head().to_string())
