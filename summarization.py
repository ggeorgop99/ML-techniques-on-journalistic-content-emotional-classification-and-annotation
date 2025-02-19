import pandas as pd
import openai
import re


# Set your OpenAI API key
openai.api_key = "your_api_key"


def preprocess_text(text):
    # Remove special characters and extra whitespace
    text = re.sub(
        r"[^\w\sΆ-Ωά-ω]", "", text
    )  # Keep Greek letters and basic punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text


def summarize_text_chunk(chunk, max_tokens=100):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are an assistant specializing in summarizing Greek text.",
                },
                {
                    "role": "user",
                    "content": f"Summarize the following text in Greek:\n{chunk}",
                },
            ],
            max_tokens=max_tokens,
            temperature=0.7,
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error: {e}"


def summarize_large_dataset(
    csv_path, text_column, output_csv_path, chunk_size=1000, max_tokens=100
):
    # Load the dataset
    df = pd.read_csv(csv_path)

    # Ensure the column exists
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in the dataset.")

    summaries = []

    for text in df[text_column]:
        # Split long text into chunks
        chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
        summarized_chunks = [
            summarize_text_chunk(chunk, max_tokens=max_tokens) for chunk in chunks
        ]

        # Combine summarized chunks
        combined_summary = " ".join(summarized_chunks)
        summaries.append(combined_summary)

    # Add summaries to a new column
    df["summary"] = summaries

    # Save the dataset with summaries
    df.to_csv(output_csv_path, index=False)
    print(f"Summarized dataset saved to {output_csv_path}")


# Example usage
csv_path = "input_dataset.csv"
output_csv_path = "summarized_dataset.csv"
summarize_large_dataset(
    csv_path, text_column="text_column_name", output_csv_path=output_csv_path
)

if __name__ == "__main__":
    greek_text = "Η Τεχνητή Νοημοσύνη αποτελεί έναν από τους πιο συναρπαστικούς τομείς της επιστήμης και της τεχνολογίας. Οι εφαρμογές της καλύπτουν διάφορους κλάδους, όπως η υγεία, οι μεταφορές και η εκπαίδευση."
    greek_text = preprocess_text(greek_text)
    summary = summarize_text_chunk(greek_text)
    print("Original Text:", greek_text)
    print("\nSummarized Text:", summary)
