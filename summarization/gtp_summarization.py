import pandas as pd
from openai import OpenAI
import re
import os

# Set OpenAI API key from environment variable
# openai.api_key = os.getenv("OPENAI_API_KEY")

api_key = "your_api key"

client = OpenAI(api_key=api_key)


def preprocess_text(text):
    """Cleans Greek text by removing unwanted characters and extra spaces."""
    if not isinstance(text, str) or pd.isna(text):
        return ""
    text = re.sub(
        r"[^\w\sΆ-Ωά-ω.,!?]", "", text
    )  # Keep Greek letters and common punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text


def summarize_text_chunk(chunk, max_tokens=100):
    """Summarizes a chunk of Greek text using OpenAI API."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an assistant summarizing Greek text.",
                },
                {
                    "role": "user",
                    "content": f"Summarize the following text in Greek:\n{chunk}",
                },
            ],
            max_tokens=max_tokens,
            temperature=0.3,  # Lower temperature for more consistency
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error: {e}"


def summarize_large_dataset(
    csv_path, text_column, output_csv_path, chunk_size=1000, max_tokens=100
):
    """Reads a CSV file, summarizes the text column, and saves the results."""
    df = pd.read_csv(csv_path)

    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in dataset.")

    summaries = []

    for text in df[text_column]:
        if pd.isna(text) or not isinstance(text, str):
            summaries.append("")
            continue

        text = preprocess_text(text)

        # Split long text into chunks while preserving sentence structure
        chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
        summarized_chunks = [
            summarize_text_chunk(chunk, max_tokens=max_tokens) for chunk in chunks
        ]

        summaries.append(" ".join(summarized_chunks))

    df["summary"] = summaries
    df.to_csv(output_csv_path, index=False)
    print(f"Summarized dataset saved to {output_csv_path}")


# Example usage
if __name__ == "__main__":
    # csv_path = "input_dataset.csv"
    # output_csv_path = "summarized_dataset.csv"
    # summarize_large_dataset(
    #     csv_path, text_column="text_column_name", output_csv_path=output_csv_path
    # )

    # Single text summary example
    greek_text = "Η Τεχνητή Νοημοσύνη αποτελεί έναν από τους πιο συναρπαστικούς τομείς της επιστήμης και της τεχνολογίας."
    greek_text = preprocess_text(greek_text)
    summary = summarize_text_chunk(greek_text)
    print("Original Text:", greek_text)
    print("\nSummarized Text:", summary)
