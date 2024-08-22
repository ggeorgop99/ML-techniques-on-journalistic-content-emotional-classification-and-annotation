import pandas as pd
import argparse
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import spacy
from nltk.tokenize import RegexpTokenizer
import numpy as np
import os
from utils import generate_unique_filename
from hunspell import Hunspell

# spacy.require_gpu()
nlp = spacy.load("el_core_news_lg")
regexp = RegexpTokenizer("\w+")

# Define the command-line arguments
parser = argparse.ArgumentParser(description="Train a sentiment analysis model.")
parser.add_argument(
    "--mode",
    type=str,
    required=True,
    choices=["bin", "nonbin"],
    help="Mode of the model: nonbin or bin",
)
parser.add_argument(
    "--file_name", type=str, required=True, help="Name of file to preprocess"
)
args = parser.parse_args()

# Access the mode arguments
mode = args.mode
file_name = args.file_name

# Load stopwords
stopwords = set(
    pd.read_csv("datasets/stopwords_greek.csv", header=None).squeeze().tolist()
)

# hspell = hunspell.HunSpell('/usr/share/hunspell/el_GR.dic', '/usr/share/hunspell/el_GR.aff')
hspell = Hunspell("el_GR")


def spell_check(text):
    corrected_text = []
    for word in text.split():
        if hspell.spell(word):
            corrected_text.append(word)
        else:
            suggestions = hspell.suggest(word)
            corrected_text.append(suggestions[0] if suggestions else word)
    return " ".join(corrected_text)


def lemmatize(text):
    doc = nlp(str(text))
    return " ".join([token.lemma_ for token in doc])


# def remove_names(text):
#     for word in text.split():
#         if word[0] == "@":
#             text = text.replace(word, "")
#     return text


def remove_names(text):
    return re.sub(r"@\w+", "", text)


def drop_numbers(text):
    return re.sub(r"\d+", "", text)


def clean_accent(text):
    accents = {
        "Ά": "Α",
        "Έ": "Ε",
        "Ί": "Ι",
        "Ή": "Η",
        "Ύ": "Υ",
        "Ό": "Ο",
        "Ώ": "Ω",
        "ά": "α",
        "έ": "ε",
        "ί": "ι",
        "ή": "η",
        "ύ": "υ",
        "ό": "ο",
        "ώ": "ω",
        "ς": "σ",
    }

    for accent, char in accents.items():
        text = text.replace(accent, char)
    return text


def preprocess_text(text, stopwords):
    text = clean_accent(text)
    text = text.lower()
    # remove urls
    text = re.sub(r"https?://\S+", "", text)
    # remove hashtags
    text = re.sub(r"#", "", text)
    text = remove_names(text)
    # remove punctuation
    text = re.sub(r"[^\w\s]", "", text)
    text = drop_numbers(text)
    tokens = regexp.tokenize(text)
    tokens = [token for token in tokens if token not in stopwords and len(token) > 3]
    return " ".join(tokens)


# Read the CSV file
df = pd.read_csv(f"datasets/{file_name}_{mode}.csv")

# Apply spellchecking
df["reviews"] = df["reviews"].apply(spell_check)

# Apply lemmatization and preprocessing
df["text_lemma"] = df["reviews"].apply(lemmatize)
df["text_proc"] = df["text_lemma"].apply(lambda x: preprocess_text(x, stopwords))

# Remove empty rows
df.dropna(subset=["text_proc"], inplace=True)
df = df[df["text_proc"].astype(bool)]

# Drop unnecessary columns
df = df.drop(columns=["text_lemma"])
df["reviews"] = df["text_proc"]
df = df.drop(columns=["text_proc"])
if "Unnamed: 2" in df.columns:
    df = df.drop(columns=["Unnamed: 2"])
df.dropna(axis=1, how="all")
print(df.head())

# Save the modified DataFrame to a new CSV file
os.makedirs("preprocessed_datasets", exist_ok=True)
unique_file_path = generate_unique_filename(
    "preprocessed_datasets", f"{file_name}", f"{mode}", "csv"
)
df.to_csv(unique_file_path, index=False)
