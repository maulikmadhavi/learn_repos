import sqlite3
import pandas as pd
import csv

conn = sqlite3.connect("favorites.db")

stu_data = pd.read_csv("favorites.csv", names=["Timestamp", "language"])

stu_data.to_sql(
    "favorites",
    conn,
    if_exists="replace",
    index=False,
    index_label=["Timestamp", "Language"],
)

# create a cursor object
cur = conn.cursor()
for row in cur.execute("SELECT * FROM favorites"):
    print(row)

conn.close()
