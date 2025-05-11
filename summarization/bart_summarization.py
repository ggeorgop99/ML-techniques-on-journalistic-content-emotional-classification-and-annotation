import pandas as pd
import re
from transformers import pipeline

MAX_INPUT_LENGTH = 512


# Preprocessing function
def preprocess_text(text):
    # Remove URLs
    text = re.sub(r"http[s]?://\S+", "", text)  # Removes http:// and https:// URLs
    text = re.sub(r"www\.\S+", "", text)  # Removes www URLs

    # Remove mentions (i.e., @username)
    text = re.sub(r"@\w+", "", text)

    # Remove hashtags (i.e., #hashtag)
    text = re.sub(r"#\w+", "", text)

    # Remove special characters but keep Greek letters, numbers, and punctuation
    text = re.sub(
        r"[^A-Za-z0-9Α-Ωα-ω\s.,;?!-]", "", text
    )  # Keeps letters, numbers, and basic punctuation

    # Remove extra whitespaces
    text = " ".join(text.split())

    return text


def summarize_greek_text(csv_file, column_name="text", max_length=200, min_length=50):
    # Load CSV file
    df = pd.read_csv(csv_file)

    # Combine all tweets into a single text
    text = " ".join(df[column_name].astype(str))

    text = preprocess_text(text)

    text = text.encode("utf-8", "ignore").decode("utf-8")

    truncated_text = text[:MAX_INPUT_LENGTH]

    # Load pre-trained summarization model
    summarizer = pipeline(
        "summarization",
        model="t5-small",
        tokenizer="t5-small",
        framework="tf",  # Ensure TensorFlow is used
        use_fast=False,  # Force slow tokenizer
    )

    # Generate summary
    summary = summarizer(
        truncated_text, max_length=max_length, min_length=min_length, do_sample=False
    )

    return summary[0]["summary_text"]


# Example usage
if __name__ == "__main__":
    input_csv = "media.csv"  # Change to your actual file name
    summary = summarize_greek_text(input_csv)
    print("Summary:")
    print(summary)
