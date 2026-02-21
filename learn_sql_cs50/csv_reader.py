import csv
from collections import Counter

# Step 1: Open the file and read line by line
with open("favorites.csv", "r") as file:
    csv_reader = csv.reader(file)
    next(csv_reader)
    for line in csv_reader:
        print(line)


# Step 2: Read the file as a dictionary
with open("favorites.csv", "r") as file:
    csv_reader = csv.DictReader(file)
    favorites = {}
    for line in csv_reader:  # Each line is now a dictionary
        favorite = line["language"]  # Access the value by key
        if favorite in favorites:
            favorites[favorite] += 1
        else:
            favorites[favorite] = 1

print(favorites)

# Step 3: Use the Counter class
with open("favorites.csv", "r") as file:
    csv_reader = csv.DictReader(file)
    favorites = Counter()  # A dictionary that keeps track of counts
    print(favorites)  # Should be empty
    for line in csv_reader:
        favorite = line["language"]
        favorites[favorite] += 1
    print(favorites)

# Step 4: Use the most_common method
with open("favorites.csv", "r") as file:
    csv_reader = csv.DictReader(file)
    favorites = Counter()
    for line in csv_reader:
        favorite = line["language"]
        favorites[favorite] += 1
    print(favorites.most_common())

# Step 5: Use sort using key as count and reverse
sorted_fav = sorted(favorites, key=favorites.get, reverse=True)
for fav in sorted_fav:
    print(f"{fav}: {favorites[fav]}")


# Step 6: Use the Counter class use lambda function
sorted_fav2 = sorted(favorites, key=lambda x: favorites[x], reverse=True)
for fav in sorted_fav2:
    print(f"{fav}: {favorites[fav]}")
