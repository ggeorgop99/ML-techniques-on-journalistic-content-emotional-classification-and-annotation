import pandas as pd

# import csv
# from itertools import zip_longest
# from difflib import SequenceMatcher
# import os
import argparse

# import sys

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from utils import generate_unique_filename


# Define the command-line arguments
parser = argparse.ArgumentParser(description="Clear the dataset.")
parser.add_argument(
    "--mode",
    type=str,
    required=True,
    choices=["bin", "nonbin"],
    help="Mode of the dataset: nonbin or bin",
)
parser.add_argument(
    "--file_name", type=str, required=True, help="Name of file to clear"
)
args = parser.parse_args()

# Access the mode arguments
mode = args.mode
file_name = args.file_name

data = pd.read_csv("dirtyreviews.csv", encoding="utf-8")
print(data.head())
data = data.drop("topic", axis=1)
data = data.drop("title", axis=1)
data = data.dropna()
data = data.drop_duplicates(subset=["comment"], keep="first")
data["stars"] = pd.to_numeric(data["stars"], errors="coerce")

# Identify rows with invalid 'stars' values
invalid_stars_rows = data[data["stars"].isna()]

# Log the rows with invalid 'stars' values
if not invalid_stars_rows.empty:
    print("Rows with invalid 'stars' values that will be dropped:")
    print(invalid_stars_rows)

data = data.dropna(subset=["stars"])
data["stars"] = data["stars"].astype(int)
print(data.head())

temp = []
temp = data["stars"].values.tolist()
name = f"{file_name}_{mode}"


if mode == "bin":
    for i in range(0, len(data["stars"])):
        if int(temp[i]) <= 3:
            temp[i] = 0
        else:
            temp[i] = 1

else:
    for i in range(0, len(data["stars"])):

        if int(temp[i]) <= 2:
            temp[i] = -1
        elif int(temp[i]) == 3:
            temp[i] = 0
        else:
            temp[i] = 1
data["stars"] = temp

cols = data.columns.tolist()
cols = cols[-1:] + cols[:-1]
data = data[cols]
data.to_csv(
    name + ".csv", header=["reviews", "sentiment"], index=False, encoding="utf-8"
)
