import pandas as pd
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
import nltk


def summarize_greek_text(csv_file, column_name="text", sentence_count=5):
    # Load CSV file
    df = pd.read_csv(csv_file)

    # Combine all tweets into a single text
    text = " ".join(df[column_name].astype(str))

    # Summarize the text
    parser = PlaintextParser.from_string(text, Tokenizer("greek"))
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, sentence_count)

    # Join summary sentences into a paragraph
    summary_text = " ".join(str(sentence) for sentence in summary)

    return summary_text


# Example usage
if __name__ == "__main__":
    nltk.download("punkt")
    input_csv = "media.csv"
    summary = summarize_greek_text(input_csv)
    print("Summary:")
    print(summary)
