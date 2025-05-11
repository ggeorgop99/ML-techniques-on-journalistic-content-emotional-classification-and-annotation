import pandas as pd
import re
import torch  # Needed for checking GPU and potentially by transformers
from transformers import (
    pipeline,
    AutoTokenizer,
)  # AutoTokenizer helps check model specifics
import math  # For calculating batches
from datetime import datetime
import os


# --- Preprocessing for Greek Text ---
def preprocess_greek_tweet(text):
    """
    Basic preprocessing for Greek tweets, suitable for transformer models.
    Removes URLs, mentions, and normalizes basic whitespace.
    """
    if not isinstance(text, str):  # Handle potential non-string data (like NaN)
        return ""

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    # Remove mentions (@username)
    text = re.sub(r"\@\w+", "", text)
    # Remove the hashtag symbol but keep the text (useful for topic context)
    text = re.sub(r"\#", "", text)
    # Normalize whitespace (replace multiple spaces/newlines with a single space)
    text = " ".join(text.split())
    # Avoid aggressive cleaning like lowercasing or punctuation removal for transformers
    # unless the specific model documentation recommends it.
    return text.strip()


# --- Data Loading ---
def load_and_preprocess_greek(csv_path, text_column="text"):
    """
    Loads Greek tweet data from a CSV file and preprocesses the text column.

    Args:
        csv_path (str): Path to the CSV file.
        text_column (str): Name of the column containing the Greek tweet text.

    Returns:
        pandas.DataFrame or None: Processed DataFrame or None if loading fails.
    """
    try:
        df = pd.read_csv(csv_path)
        # df = df.head(5000)  # Limit to first 5000 rows for performance
        if text_column not in df.columns:
            raise ValueError(
                f"Text column '{text_column}' not found in CSV file '{csv_path}'."
            )

        print(f"Original number of tweets: {len(df)}")
        # Drop rows where the target text column is missing
        df.dropna(subset=[text_column], inplace=True)
        print(f"Number of tweets after dropping NA in '{text_column}': {len(df)}")

        # Apply preprocessing
        df["cleaned_text"] = df[text_column].apply(preprocess_greek_tweet)

        # Filter out rows that became empty after preprocessing
        df = df[df["cleaned_text"].str.strip() != ""]
        print(f"Number of tweets after preprocessing and removing empty: {len(df)}")

        if df.empty:
            print(
                f"Warning: No valid text data found in column '{text_column}' after preprocessing."
            )

        return df

    except FileNotFoundError:
        print(f"Error: CSV file not found at '{csv_path}'")
        return None
    except ValueError as ve:
        print(f"Data loading error: {ve}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during data loading/preprocessing: {e}")
        return None


# --- Abstractive Summarization for Greek ---
def summarize_tweets_transformer_greek(
    df,
    model_name="nlpaueb/grt5-small",  # Greek T5 model (small version)
    max_chunk_length_words=400,  # Safer limit for T5 (often 512 tokens)
    summary_max_length=150,  # Max tokens in output summary
    summary_min_length=30,  # Min tokens in output summary
    batch_size=8,
    s1_summary_max_length_factor=1.5,  # S1 chunks can be a bit longer
):  # Adjust based on GPU memory
    """
    Summarizes Greek text from a DataFrame using a Hugging Face transformer model.

    Args:
        df (pd.DataFrame): DataFrame with a 'cleaned_text' column containing Greek text.
        model_name (str): Name of the Greek summarization model from Hugging Face Hub.
        max_chunk_length_words (int): Approximate maximum words per chunk for processing.
                                      Token count is more accurate but harder to estimate beforehand.
        summary_max_length (int): Maximum number of tokens for the generated summary.
        summary_min_length (int): Minimum number of tokens for the generated summary.
        batch_size (int): How many chunks to process simultaneously (requires GPU).
        s1_summary_max_length_factor (float): Factor to increase max length for S1 chunks.

    Returns:
        str: The generated summary, or an error message string.
    """
    # Combine all cleaned tweets into a single text block
    # Using ". " as a separator helps the model recognize sentence boundaries.
    all_text_S1 = ". ".join(df["cleaned_text"].astype(str).tolist())
    # Consolidate multiple periods and ensure space after period
    all_text_S1 = re.sub(r"\.+", ".", all_text_S1)
    all_text_S1 = re.sub(r"\s*\.\s*", ". ", all_text_S1).strip()

    if not all_text_S1:
        return "No text available to summarize after preprocessing."

    print(f"\nInitializing summarization pipeline with model: {model_name}...")
    try:
        # Check for GPU availability
        device = 0 if torch.cuda.is_available() else -1
        if device == 0:
            print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        else:
            print("No GPU detected, using CPU (summarization might be slow).")

        # Load the summarization pipeline
        # Explicitly providing the tokenizer ensures consistency
        summarizer = pipeline(
            "summarization",
            model=model_name,
            tokenizer=model_name,  # Use the tokenizer associated with the model
            device=device,
        )

        print("Pipeline loaded.")

        # --- STAGE 1: Summarize original tweet chunks ---
        print("\n--- Starting Stage 1 Summarization (Tweets to S1 Summaries) ---")
        words_S1 = all_text_S1.split()
        chunks_S1 = [
            " ".join(words_S1[i : i + max_chunk_length_words])
            for i in range(0, len(words_S1), max_chunk_length_words)
        ]
        print(
            f"Stage 1: Input text ({len(words_S1)} words) split into {len(chunks_S1)} chunk(s)."
        )

        if not chunks_S1:
            return "Stage 1: Text could not be split into chunks."

        summaries_S1 = []
        num_batches_S1 = math.ceil(len(chunks_S1) / batch_size)
        # For S1 summaries, we can allow them to be a bit longer than the final target,
        # as they will be inputs to S2.
        s1_max_len = int(summary_max_length * s1_summary_max_length_factor)
        s1_min_len = int(
            summary_min_length * s1_summary_max_length_factor * 0.8
        )  # scale min too

        for i in range(num_batches_S1):
            batch_chunks_S1 = chunks_S1[i * batch_size : (i + 1) * batch_size]
            print(
                f"Stage 1: Processing batch {i+1}/{num_batches_S1} ({len(batch_chunks_S1)} chunks)"
            )
            try:
                chunk_summaries_S1_result = summarizer(
                    batch_chunks_S1,
                    max_length=s1_max_len,
                    min_length=s1_min_len,
                    do_sample=False,
                    truncation=True,
                )
                summaries_S1.extend(
                    [s["summary_text"] for s in chunk_summaries_S1_result]
                )
            except Exception as batch_error_s1:
                print(
                    f"Error processing Stage 1 batch {i+1}: {batch_error_s1}. Skipping this batch."
                )
                summaries_S1.extend(["[S1 Error]" for _ in batch_chunks_S1])

        intermediate_long_summary = " ".join(
            s for s in summaries_S1 if s != "[S1 Error]"
        ).strip()

        if not intermediate_long_summary:
            return "No summary generated after Stage 1 (all chunks might have failed or were empty)."
        print(
            f"--- Stage 1 Summarization Complete. Intermediate summary length: {len(intermediate_long_summary.split())} words ---"
        )

        # --- STAGE 2: Decide if intermediate_long_summary needs further summarization ---
        needs_stage2 = False
        # Condition: if ILS is longer than one processing chunk AND it was formed from multiple S1 summaries
        if (
            len(intermediate_long_summary.split()) > (max_chunk_length_words * 1.1)
            and len(summaries_S1) > 1
        ):
            needs_stage2 = True

        if needs_stage2:
            print(
                "\n--- Starting Stage 2 Summarization (Intermediate Summary to Final Summary) ---"
            )
            words_S2_input = intermediate_long_summary.split()
            chunks_S2 = [
                " ".join(words_S2_input[i : i + max_chunk_length_words])
                for i in range(0, len(words_S2_input), max_chunk_length_words)
            ]
            print(
                f"Stage 2: Intermediate summary ({len(words_S2_input)} words) split into {len(chunks_S2)} chunk(s)."
            )

            if not chunks_S2:
                print(
                    "Stage 2: Could not split intermediate summary into chunks. Returning Stage 1 result."
                )
                return intermediate_long_summary

            summaries_S2 = []
            num_batches_S2 = math.ceil(len(chunks_S2) / batch_size)
            for i in range(num_batches_S2):
                batch_chunks_S2 = chunks_S2[i * batch_size : (i + 1) * batch_size]
                print(
                    f"Stage 2: Processing batch {i+1}/{num_batches_S2} ({len(batch_chunks_S2)} chunks)"
                )
                try:
                    # Use the target summary_max_length and summary_min_length for this final stage
                    chunk_summaries_S2_result = summarizer(
                        batch_chunks_S2,
                        max_length=summary_max_length,
                        min_length=summary_min_length,
                        do_sample=False,
                        truncation=True,
                    )
                    summaries_S2.extend(
                        [s["summary_text"] for s in chunk_summaries_S2_result]
                    )
                except Exception as batch_error_s2:
                    print(
                        f"Error processing Stage 2 batch {i+1}: {batch_error_s2}. Skipping this batch."
                    )
                    summaries_S2.extend(["[S2 Error]" for _ in batch_chunks_S2])

            final_summary = " ".join(
                s for s in summaries_S2 if s != "[S2 Error]"
            ).strip()

            if not final_summary:
                print(
                    "Warning: Stage 2 summarization produced an empty result. Falling back to Stage 1 result."
                )
                return intermediate_long_summary

            print(
                f"--- Stage 2 Summarization Complete. Final summary length: {len(final_summary.split())} words ---"
            )
        else:
            print(
                "\nIntermediate summary is concise enough or from a single S1 chunk. Stage 2 summarization skipped."
            )
            final_summary = intermediate_long_summary

        return final_summary

    except ImportError as ie:
        print(
            f"Import Error: {ie}. Make sure 'transformers', 'torch', and 'pandas' are installed."
        )
        return f"Import Error: {ie}"
    except Exception as e:
        # Catch potential errors during pipeline loading or processing
        error_message = str(e).lower()
        if "out of memory" in error_message:
            print("\nCUDA Out Of Memory Error!")
            print("Try reducing 'batch_size' or 'max_chunk_length_words'.")
            return (
                "Error: CUDA Out Of Memory. Reduce batch_size/max_chunk_length_words."
            )
        elif "can't load" in error_message or "not found" in error_message:
            print(f"\nError loading model '{model_name}'.")
            print("Please check:")
            print("  1. The model name is correct.")
            print("  2. You have an internet connection to download the model.")
            print(
                "  3. The model exists on the Hugging Face Hub (huggingface.co/models)."
            )
            return f"Error: Could not load model '{model_name}'. Check name and connection."
        else:
            print(f"\nAn unexpected error occurred during summarization: {e}")
            # Consider logging the full traceback here for debugging
            # import traceback; traceback.print_exc()
            return f"Error during abstractive summarization: {e}"


# --- Function to Save Summary ---
def save_summary_to_file(summary_text, input_csv_path, model_name, output_dir="."):
    """
    Saves the generated summary and metadata to a text file.

    Args:
        summary_text (str): The summary string generated by the model.
        input_csv_path (str): Path to the original input CSV file.
        model_name (str): Name of the summarization model used.
        output_dir (str): Directory where the summary file will be saved (default: current directory).
    """
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # 1. Get current timestamp
        now = datetime.now()
        timestamp_str = now.strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS
        readable_timestamp_str = now.strftime(
            "%Y-%m-%d %H:%M:%S"
        )  # Format for metadata

        # 2. Create descriptive filename
        base_input_name = os.path.basename(input_csv_path)  # Get filename from path
        file_name_root, _ = os.path.splitext(base_input_name)  # Remove .csv extension
        output_filename = f"{file_name_root}_summary_{timestamp_str}.txt"
        output_filepath = os.path.join(output_dir, output_filename)

        # 3. Prepare content with metadata
        file_content = f"""--- Summary Report ---
            Generation Date: {readable_timestamp_str}
            Input Data File: {input_csv_path}
            Summarization Model: {model_name}

            --- Summary Text ---
            {summary_text}
"""
        # 4. Write to file (using UTF-8 encoding for Greek characters)
        with open(output_filepath, "w", encoding="utf-8") as f:
            f.write(file_content)

        print(f"\n✅ Summary successfully saved to:")
        print(f"   {output_filepath}")

    except IOError as e:
        print(f"\n❌ Error: Could not write summary to file.")
        print(f"   Reason: {e}")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred during saving: {e}")


# --- Main Execution Logic ---
if __name__ == "__main__":
    # --- Configuration ---
    CSV_FILE = "media.csv"
    TEXT_COLUMN = "text"

    # Model Options:
    # "IMISLab/GreekT5-mt5-small-greeksum" (Faster, less memory)
    # "IMISLab/GreekT5-umt5-base-greeksum" (Potentially more accurate, slower, more memory)
    GREEK_MODEL_NAME = "IMISLab/GreekT5-mt5-small-greeksum"
    OUTPUT_DIRECTORY = "summaries_hierarchical"

    # Summarization Parameters
    MAX_WORDS_PER_CHUNK = 400  # Input chunk size (adjust based on model and memory)
    TARGET_SUMMARY_MAX_TOKENS = 141  # Desired max summary length
    TARGET_SUMMARY_MIN_TOKENS = 40  # Desired min summary length
    PROCESSING_BATCH_SIZE = (
        8  # Num chunks per batch (reduce if GPU memory errors occur)
    )
    S1_SUMMARY_MAX_LENGTH_FACTOR = 1.5

    # --- End Configuration ---

    print("Starting Greek Tweet Summarization Process...")
    print(f"Using model: {GREEK_MODEL_NAME}")
    print(f"Input file: {CSV_FILE} (Column: '{TEXT_COLUMN}')")

    # 1. Load and Preprocess Data
    tweet_df_greek = load_and_preprocess_greek(CSV_FILE, text_column=TEXT_COLUMN)

    # 2. Check if data loading was successful and data is available
    if tweet_df_greek is not None and not tweet_df_greek.empty:
        print("\nData loaded and preprocessed successfully.")

        # 3. Perform Summarization
        summary_result = summarize_tweets_transformer_greek(
            tweet_df_greek,
            model_name=GREEK_MODEL_NAME,
            max_chunk_length_words=MAX_WORDS_PER_CHUNK,
            summary_max_length=TARGET_SUMMARY_MAX_TOKENS,  # This is for the S2 (final) stage
            summary_min_length=TARGET_SUMMARY_MIN_TOKENS,  # This is for the S2 (final) stage
            batch_size=PROCESSING_BATCH_SIZE,
            s1_summary_max_length_factor=S1_SUMMARY_MAX_LENGTH_FACTOR,  # New parameter
        )

        print("\n--- Generated Summary ---")
        if (
            summary_result is None
            or "Error:" in summary_result
            or not summary_result.strip()
        ):
            print(
                f"Summarization failed or produced an empty/error result: {summary_result}"
            )
        else:
            print(summary_result)
            print("-------------------------")
            save_summary_to_file(
                summary_text=summary_result,
                input_csv_path=CSV_FILE,
                model_name=GREEK_MODEL_NAME,
                output_dir=OUTPUT_DIRECTORY,
            )

    elif tweet_df_greek is not None and tweet_df_greek.empty:
        print(
            "\nProcessing stopped: No valid tweet data remained after loading and preprocessing."
        )
    else:
        # Error message was already printed during loading
        print("\nProcessing stopped due to errors during data loading.")

    print("\nSummarization process finished.")
