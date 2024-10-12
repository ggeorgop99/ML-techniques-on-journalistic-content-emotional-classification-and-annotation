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


# Function to plot sentiment distribution
def plot_sentiment_distribution(Y, model_dir, file_name, mode):
    unique_sentiments, counts = np.unique(Y, return_counts=True)
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=unique_sentiments, y=counts, palette="viridis")
    plt.title("Distribution of Sentiments", fontsize=16)
    plt.xlabel("Sentiment", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    for i, count in enumerate(counts):
        ax.text(i, count, str(count), ha="center", va="bottom", fontsize=12)

    plt.savefig(
        f"{model_dir}/{file_name}_sentiment_distribution_{mode}.png"
    )  # Save the plot
    plt.show()  # Show the plot


# Function to plot and save graphs
def plot_and_save(history, metric, model_dir, file_name, mode):
    plt.figure()
    plt.plot(history[metric], label=f"Train {metric.capitalize()}")
    plt.plot(history[f"val_{metric}"], label=f"Validation {metric.capitalize()}")
    if f"test_{metric}" in history:
        plt.plot(
            history[f"test_{metric}"],
            label=f"Test {metric.capitalize()}",
            linestyle="--",
        )

    plt.title(f"Model {metric.capitalize()}")
    plt.ylabel(metric.capitalize())
    plt.xlabel("Epoch")
    plt.legend(loc="upper left")
    plt.savefig(f"{model_dir}/{file_name}_{metric}_{mode}.png")  # Save the plot
    plt.show()  # Show the plot
    plt.close()  # Close the figure


# Define command-line arguments
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
dir_path = f"savedmodel_{mode}"
model_dir = f"{dir_path}/{file_name}_model"  # Directory inside savedmodel_bin for this dataset/model
model_path = f"{model_dir}/{file_name}_{mode}"

# Create directories
os.makedirs(model_dir, exist_ok=True)

dataset_path = f"preprocessed_datasets/{file_name}_{mode}.csv"

# CSV file to save results
results_csv_path = "savedmodel_bin/model_results.csv"

# Initialize results DataFrame if the file doesn't exist
if not os.path.exists(results_csv_path):
    results_df = pd.DataFrame(
        columns=["Model Name", "Loss", "Accuracy", "Precision", "Recall", "AUC", "MSE"]
    )
    results_df.to_csv(results_csv_path, index=False)

# Load dataset
df = pd.read_csv(dataset_path, usecols=[0, 1])
# Sort for easier normalization
df = df.sort_values(by=["sentiment"], ascending=True)

X = df["reviews"].values
Y = df["sentiment"].values

# Plot sentiment distribution
plot_sentiment_distribution(Y, model_dir, file_name, mode)

if mode == "nonbin":
    N = len(df.loc[df["sentiment"] == -1])
    Z = len(df.loc[df["sentiment"] == 0])
    P = len(df.loc[df["sentiment"] == 1])
    Z = min(Z, N)
    length1 = P - Z
    length2 = N - Z
    X = X[length2:-length1]
    Y = Y[length2:-length1]
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

# Split into train/valid/test: 70% training, 15% validation, 15% testing
X_train, X_temp, Y_train, Y_temp = train_test_split(
    X, Y, test_size=0.3, random_state=42
)
X_valid, X_test, Y_valid, Y_test = train_test_split(
    X_temp, Y_temp, test_size=0.5, random_state=42
)

vec = CountVectorizer()
vec.fit(X_train.astype("U"))

# Save the vectorizer inside the model directory
with open(f"{model_dir}/count_vectorizer_{file_name}_{mode}.pkl", "wb") as f:
    pickle.dump(vec, f)

x_train = vec.transform(X_train.astype("U"))
x_valid = vec.transform(X_valid.astype("U"))
x_test = vec.transform(X_test.astype("U"))

if mode == "nonbin":
    Y_train += 1  # Adjust labels for non-binary
    Y_valid += 1
    Y_test += 1
    Y_train = to_categorical(Y_train)
    Y_valid = to_categorical(Y_valid)
    Y_test = to_categorical(Y_test)

# Build the model
model = Sequential(
    [
        Dense(16, input_dim=x_train.shape[1], activation="relu"),
        Dropout(0.5),  # Add dropout for regularization
        Dense(16, activation="relu"),
        Dropout(0.5),  # Add dropout for regularization
        Dense(outp_node, activation="sigmoid"),
    ]
)

# Compile the model with additional metrics
model.compile(
    loss=loss_func,
    optimizer=Adam(),
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall(),
        tf.keras.metrics.AUC(),
        tf.keras.metrics.MeanSquaredError(),
    ],
)

model.summary()

# Early stopping
early_stopping = EarlyStopping(monitor="val_auc", patience=5, restore_best_weights=True)

# Train the model
history = model.fit(
    x_train,
    Y_train,
    validation_data=(x_valid, Y_valid),
    epochs=100,
    verbose=True,
    batch_size=16,
    callbacks=[early_stopping],
)

# Evaluate the model on the test set and add results to history

test_loss, test_accuracy, test_precision, test_recall, test_auc, test_mse = (
    model.evaluate(x_test, Y_test)
)

# Ensure the AUC key exists in history
if "AUC" in history.history:
    history.history["test_auc"] = [test_auc] * len(history.history["AUC"])
else:
    history.history["test_auc"] = [test_auc] * len(
        history.history["accuracy"]
    )  # Use a different metric length

# Save the model inside the model directory
model.save(f"{model_path}.h5")
model.save(f"{model_path}.keras")
np.save(f"{model_path}_history.npy", history.history)

# Save training graphs with test results
plot_and_save(history.history, "accuracy", model_dir, file_name, mode)
plot_and_save(history.history, "loss", model_dir, file_name, mode)
plot_and_save(history.history, "precision", model_dir, file_name, mode)
plot_and_save(history.history, "recall", model_dir, file_name, mode)
plot_and_save(history.history, "auc", model_dir, file_name, mode)
plot_and_save(history.history, "mean_squared_error", model_dir, file_name, mode)


# Save model results to a CSV
results = {
    "Model Name": f"{file_name}_{mode}",
    "Loss": test_loss,
    "Accuracy": test_accuracy,
    "Precision": test_precision,
    "Recall": test_recall,
    "AUC": test_auc,
    "MSE": test_mse,
}

results_df = pd.DataFrame([results])
results_df.to_csv(
    results_csv_path, mode="a", header=False, index=False
)  # Append results

# Plot model architecture
plot_model(
    model,
    to_file=f"{model_dir}/model_plot_{file_name}_{mode}.png",
    show_shapes=True,
    show_layer_names=True,
)
