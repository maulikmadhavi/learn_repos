import pandas as pd

# Series = A Pandas 1-dimensional labeled array that can hold any data type.
# Think of it like a column in a spreadsheet (1-dimensional).

data = [100, 102, 104.2, ] # float
# data = [True, False, True] # boolean
# data = ['apple', 'banana', 'cherry'] # string
# data = {'a': 100, 'b': 102, 'c': 104.2} # dictionary <--- No need to specify index here
series = pd.Series(data)
series = pd.Series(data, index=['a', 'b', 'c']) # custom index
# number of elements in the index must be the same as the number of elements in data

print(type(series))  # <class 'pandas.core.series.Series'>
print(series)

# left side is the index, right side is the value
# at the end it shows dtype of the data in the series
# It can be int, float, string, bool, object, etc.

# properties and attributes
print("Series values:", series.values)  # numpy array of values
print("Series index:", series.index)    # index range or custom index
print("Series dtype:", series.dtype)    # data type of the series
print("Series shape:", series.shape)    # shape of the series (number of elements)
print("Series size:", series.size)      # number of elements in the series

print("Series loc['a']:", series.loc['a'])  # access value by label/index

print("Series iloc[0]:", series.iloc[0])  # access value by position


# Filter

series_filtered = series[series > 101]  # filter values greater than 101
print("Filtered Series (values > 101):", series_filtered)