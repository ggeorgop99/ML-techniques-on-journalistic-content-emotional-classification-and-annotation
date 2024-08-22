# may need to do pip install pydot-ng and sudo apt install graphvizfor it to work

import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import argparse
import os


def plot_sentiment_distribution(Y):
    unique_sentiments, counts = np.unique(Y, return_counts=True)
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=unique_sentiments, y=counts, palette="viridis")
    plt.title("Distribution of Sentiments", fontsize=16)
    plt.xlabel("Sentiment", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    for i, count in enumerate(counts):
        ax.text(i, count, str(count), ha="center", va="bottom", fontsize=12)
    plt.show()


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
    "--mode1",
    type=str,
    required=True,
    choices=["train", "graphs"],
    help="Operation mode: train or graphs",
)
parser.add_argument(
    "--file_name", type=str, required=True, help="Name of file to preprocess"
)
args = parser.parse_args()

# Access the mode arguments
mode = args.mode
mode1 = args.mode1
file_name = args.file_name
dir_path = f"savedmodel_{mode}"
model_path = f"{dir_path}/savedmodel_{file_name}_{mode}"
dataset_path = f"preprocessed_datasets/{file_name}_{mode}.csv"

if mode1 == "train":
    dataset = f"{dataset_path}"
    df = pd.read_csv(dataset, usecols=[0, 1])
    # sort for easier normalisation
    df = df.sort_values(by=["sentiment"], ascending=True)

    X = df["reviews"].values
    Y = df["sentiment"].values

    # script to normalize sentiment values to be equal
    # pos and neg sentiment are counted and their diff
    # is removed from X in pos reviews

    plot_sentiment_distribution(Y)

    if mode == "nonbin":
        N = len(df.loc[df["sentiment"] == -1])
        Z = len(df.loc[df["sentiment"] == 0])
        P = len(df.loc[df["sentiment"] == 1])
        Z = min(Z, N)
        length1 = P - Z
        length2 = N - Z
        X = X[length2:-length1]
        print(X)
        Y = Y[length2:-length1]
        print(Y)
        outp_node = 3
        loss_func = "categorical_crossentropy"
    else:
        Z = len(df.loc[df["sentiment"] == 0])
        P = len(df.loc[df["sentiment"] == 1])
        length = abs(Z - P)
        if P > Z:
            X = X[:-length]
            Y = Y[:-length]
        else:
            X = X[length:]
            Y = Y[length:]
        outp_node = 1
        loss_func = "binary_crossentropy"

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.3, random_state=42
    )
    vec = CountVectorizer()
    vec.fit(X_train.astype("U"))

    os.makedirs(f"{dir_path}", exist_ok=True)
    with open(f"{dir_path}/count_vectorizer_{file_name}_{mode}.pkl", "wb") as f:
        pickle.dump(vec, f)

    x_train = vec.transform(X_train.astype("U"))
    x_test = vec.transform(X_test.astype("U"))

    if mode == "nonbin":
        Y_train += 1  # because it does not like -1
        Y_test += 1  # because it does not like -1
        Y_train = to_categorical(Y_train)  # 3 classes to categories for keras
        Y_test = to_categorical(Y_test)

    model = Sequential(
        [
            Dense(16, input_dim=x_train.shape[1], activation="relu"),
            Dropout(0.5),  # Add dropout for regularization
            Dense(16, activation="relu"),
            Dropout(0.5),  # Add dropout for regularization
            Dense(outp_node, activation="sigmoid"),
        ]
    )
    model.compile(
        loss=loss_func,
        optimizer=Adam(),
        metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
    )
    model.summary()

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    history = model.fit(
        x_train,
        Y_train,
        validation_data=(x_test, Y_test),
        epochs=100,
        verbose=True,
        batch_size=16,
        callbacks=[early_stopping],
    )

    model.save(f"{model_path}.h5")
    model.save(f"{model_path}.keras")
    np.save(f"{model_path}.npy", history.history)

if mode1 == "graphs":
    model = tf.keras.models.load_model(f"{model_path}.keras")
    history = np.load(f"{model_path}.npy", allow_pickle=True).item()

    plot_model(
        model, to_file=f"model_plot_{mode}.png", show_shapes=True, show_layer_names=True
    )

    # list all data in history

    plt.plot(history["accuracy"], label="Train Accuracy")
    plt.plot(history["val_accuracy"], label="Validation Accuracy")
    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(loc="upper left")
    plt.show()

    plt.plot(history["loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.title("Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(loc="upper left")
    plt.show()

    plt.plot(history["precision"], label="Train Precision")
    plt.plot(history["val_precision"], label="Validation Precision")
    plt.title("Model Precision")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(loc="upper left")
    plt.show()

    plt.plot(history["recall"], label="Train Recall")
    plt.plot(history["val_recall"], label="Validation Recall")
    plt.title("Model Recall")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(loc="upper left")
    plt.show()
