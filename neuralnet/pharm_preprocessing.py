import pandas as pd

# df = pd.read_csv("datasets/pharm_translated_greek.csv")

df = pd.read_csv("datasets/pharm_small_test.csv")

# df = df[["text", "sentiment"]]

df = df[["text", "hate"]]
df = df.rename(columns={"text": "reviews"})
df = df.rename(columns={"hate": "sentiment"})


sentiment_mapping_bin = {"negative": 0, "neutral": 1, "positive": 1}

# sentiment_mapping_nonbin = {"negative": -1, "neutral": 0, "positive": 1}

hate_to_sentiment_mapping = {1: 0, 0: 1}

sentiment_mapping = sentiment_mapping_bin

# if mode == "bin" else sentiment_mapping_nonbin

# df["sentiment"] = df["sentiment"].map(sentiment_mapping)

df["sentiment"] = df["sentiment"].map(hate_to_sentiment_mapping)


df.to_csv("datasets/pharm_small_test_bin.csv", index=False)
