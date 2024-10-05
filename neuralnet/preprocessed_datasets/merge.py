import pandas as pd


def merge_csv(*files):
    dfs = []

    for file in files:
        if file:
            df = pd.read_csv(file)
            dfs.append(df)
            print(f"\nDataFrame from {file}:")
            print(df.head())

    merged_df = pd.concat(dfs, ignore_index=True)

    merged_df.drop_duplicates(inplace=True)

    print("\nMerged DataFrame:")
    print(merged_df.head())

    merged_df.to_csv("targetAndPharm30Spellchecked.csv", index=False)


file1 = "targetSpellchecked_bin.csv"
file2 = "pharmSpellchecked30_bin.csv"
# file3 = "dataset3Spellchecked_bin.csv"
# file4 = "dataset4Spellchecked_bin.csv"

merge_csv(file1, file2)
