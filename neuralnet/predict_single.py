import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
import argparse
import os
from preprocessing import preprocess_text, clean_accent, spell_check, lemmatize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import spacy
from hunspell import Hunspell

# Load required models and tools
nlp = spacy.load("el_core_news_lg")
regexp = RegexpTokenizer("\w+")
hspell = Hunspell("el_GR")
stopwords = set(pd.read_csv("datasets/stopwords_greek.csv", header=None).squeeze().tolist())

def load_model_and_vectorizer(model_dir, file_name, mode):
    # Load the model
    model_path = f"{model_dir}/{file_name}_{mode}.h5"
    model = load_model(model_path)
    
    # Load the vectorizer
    vectorizer_path = f"{model_dir}/count_vectorizer_{file_name}_{mode}.pkl"
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    
    return model, vectorizer

def preprocess_single_text(text):
    # Apply the same preprocessing steps as in preprocessing.py
    text = spell_check(text)
    text = lemmatize(text)
    text = preprocess_text(text, stopwords)
    return text

def predict_sentiment(text, model, vectorizer, mode):
    # Transform the text using the vectorizer
    x = vectorizer.transform([text])
    
    # Get prediction
    probability = model.predict(x)[0]
    
    if mode == "bin":
        # For binary classification
        prediction = 1 if probability > 0.5 else 0
        return prediction, probability
    else:
        # For non-binary classification
        prediction = np.argmax(probability) - 1  # Convert back to -1, 0, 1
        return prediction, probability

def main():
    parser = argparse.ArgumentParser(description="Predict sentiment for a single text input")
    parser.add_argument("--mode", type=str, required=True, choices=["bin", "nonbin"], 
                      help="Mode of the model: nonbin or bin")
    parser.add_argument("--model_name", type=str, required=True, 
                      help="Name of the trained model to use")
    parser.add_argument("--text", type=str, required=True, 
                      help="Text to analyze")
    args = parser.parse_args()
    
    # Set up paths
    model_dir = f"savedmodel_{args.mode}/{args.model_name}_model"
    
    # Load model and vectorizer
    model, vectorizer = load_model_and_vectorizer(model_dir, args.model_name, args.mode)
    
    # Preprocess the text
    preprocessed_text = preprocess_single_text(args.text)
    
    # Get prediction
    prediction, probability = predict_sentiment(preprocessed_text, model, vectorizer, args.mode)
    
    # Print results
    print("\nResults:")
    print(f"Original text: {args.text}")
    print(f"Preprocessed text: {preprocessed_text}")
    print(f"Predicted sentiment: {prediction}")
    
    if args.mode == "bin":
        print(f"Probability of positive sentiment: {probability[0]:.4f}")
        sentiment = "Positive" if prediction == 1 else "Negative"
    else:
        print(f"Probabilities: Negative: {probability[0]:.4f}, Neutral: {probability[1]:.4f}, Positive: {probability[2]:.4f}")
        sentiment = "Negative" if prediction == -1 else "Neutral" if prediction == 0 else "Positive"
    
    print(f"Final sentiment: {sentiment}")

if __name__ == "__main__":
    main() 