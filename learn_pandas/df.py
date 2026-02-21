import pandas as pd

# DataFrame = A tabular data structure with rows AND columns. (2 Dimensional)
# Think of it like a spreadsheet.

data = {
    "Name": ["Spongebob", "Patrick", "Squidward"],
    "Age" : [35, 40, 52],
}


df = pd.DataFrame(data)
print("original DataFrame:")
print(df)

# Change the index:
df = pd.DataFrame(data, index=["Employee-1", "Employee-2", "Employee-3"])
print("DataFrame with custom index:")
print(df)


# Access
print("Accessing data using loc:")
print(df.loc["Employee-2"])  # Access row by label/index

print("Accessing data using iloc:")
print(df.iloc[1])  # Access row by position

# Add element 
df.loc["Employee-4"] = ["Sandy", 30]
print("DataFrame after adding Employee-4:")
print(df)

# Add new column of "department"
df["Department"] = ["HR", "Finance", "N/A", "Operations"]
print("DataFrame after adding Department column:")
print(df)

# Add new element with dictionary
new_data = {
    "Name": "Mr. Krabs",
    "Age": 55,
    "Department": "Management"
}
df = pd.concat((df, pd.DataFrame(new_data, index=["Emp-5"])))
print("DataFrame after adding Employee-5:")
print(df)


# 
df2 = pd.DataFrame([{"player": "bob", "age": 33},{"player": "alex", "age": 36}, ])
print(df2)

df2=pd.concat([df2, pd.DataFrame(
    [{"player":"chagan", "age":22}]
                                )])
print(df2)