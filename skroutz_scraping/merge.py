import pandas as pd

file1 = "dataset1_dirtyreviews.csv"
file2 = "dataset2_dirtyreviews.csv"
file3 = "dataset3_dirtyreviews.csv"
file4 = "dataset4_dirtyreviews.csv"


df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)
df3 = pd.read_csv(file3)
df4 = pd.read_csv(file4)

dfs = [df1, df2, df3, df4]

print("First DataFrame:")
print(df1.head())
print("\nSecond DataFrame:")
print(df2.head())
print("\nThird DataFrame:")
print(df3.head())
print("\nFourth DataFrame:")
print(df4.head())

merged_df = pd.concat(dfs, ignore_index=True)
merged_df.drop_duplicates(inplace=True)
print("\nMerged DataFrame:")
print(merged_df.head())

merged_df.to_csv("dataset_dirtyreviews.csv", index=False)
