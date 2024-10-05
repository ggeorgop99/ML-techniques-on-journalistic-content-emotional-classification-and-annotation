import pandas as pd


def split_csv(file_path, output_file1, output_file2):
    df = pd.read_csv(file_path)

    total_rows = len(df)

    split_index = int(len(df) * 0.7)

    df1 = df.iloc[:split_index]
    df2 = df.iloc[split_index:]

    df1.to_csv(output_file1, index=False)
    df2.to_csv(output_file2, index=False)

    print(f"CSV file successfully split into {output_file1} and {output_file2}")


file_path = "pharmSpellchecked_bin.csv"
output_file1 = f"{file_path}_70.csv"
output_file2 = f"{file_path}_30.csv"

split_csv(file_path, output_file1, output_file2)
