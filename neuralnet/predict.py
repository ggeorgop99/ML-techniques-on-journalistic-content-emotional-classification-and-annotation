import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
import argparse
import os

def load_model_and_vectorizer(model_dir, file_name, mode):
    # Load the model
    model_path = f"{model_dir}/{file_name}_{mode}.h5"
    model = load_model(model_path)
    
    # Load the vectorizer
    vectorizer_path = f"{model_dir}/count_vectorizer_{file_name}_{mode}.pkl"
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    
    return model, vectorizer

def predict_sentiment(texts, model, vectorizer, mode):
    # Transform the texts using the vectorizer
    x = vectorizer.transform(texts.astype('U'))
    
    # Get predictions
    probabilities = model.predict(x)
    
    if mode == "bin":
        # For binary classification, convert probabilities to binary predictions
        predictions = (probabilities > 0.5).astype(int)
        return predictions.flatten(), probabilities.flatten()
    else:
        # For non-binary classification, get the class with highest probability
        predictions = np.argmax(probabilities, axis=1) - 1  # Convert back to -1, 0, 1
        return predictions, probabilities

def main():
    parser = argparse.ArgumentParser(description="Predict sentiment for unlabeled text data using trained neural network models")
    parser.add_argument("--mode", type=str, required=True, choices=["bin", "nonbin"], help="Mode of the model: nonbin or bin")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the trained model to use")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--text_column", type=str, required=True, help="Name of the column containing text to analyze")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save results")
    args = parser.parse_args()
    
    # Set up paths
    model_dir = f"savedmodel_{args.mode}/{args.model_name}_model"
    
    # Load model and vectorizer
    model, vectorizer = load_model_and_vectorizer(model_dir, args.model_name, args.mode)
    
    # Read input file
    df = pd.read_csv(args.input_file)
    
    # Get predictions
    predictions, probabilities = predict_sentiment(df[args.text_column].values, model, vectorizer, args.mode)
    
    # Add predictions to dataframe
    df['predicted_sentiment'] = predictions
    df['sentiment_probability'] = probabilities
    
    # Save results
    df.to_csv(args.output_file, index=False)
    print(f"Results saved to {args.output_file}")
    
    # Print summary
    sentiment_counts = df['predicted_sentiment'].value_counts()
    print("\nSentiment Distribution:")
    for sentiment, count in sentiment_counts.items():
        print(f"Sentiment {sentiment}: {count} texts")

if __name__ == "__main__":
    main() 