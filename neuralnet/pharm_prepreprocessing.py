if file_name == "pharm":
    df = df[["text", "sentiment"]]
    df = df.rename(columns={"text": "reviews"})

    sentiment_mapping_bin = {"negative": 0, "neutral": 1, "positive": 1}
    sentiment_mapping_nonbin = {"negative": -1, "neutral": 0, "positive": 1}
    sentiment_mapping = (
        sentiment_mapping_bin if mode == "bin" else sentiment_mapping_nonbin
    )
    df["sentiment"] = df["sentiment"].map(sentiment_mapping)
