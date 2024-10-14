import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    log_loss,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)
import argparse
import os


# Save plots instead of showing them
def save_plot(fig, filename):
    fig.savefig(filename)
    plt.close(fig)


def plot_sentiment_distribution(Y_test, file_name, mode):
    unique_sentiments, counts = np.unique(Y_test, return_counts=True)
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=unique_sentiments, y=counts, palette="viridis")
    plt.title("Distribution of Sentiments", fontsize=16)
    plt.xlabel("Sentiment", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    for i, count in enumerate(counts):
        ax.text(i, count, str(count), ha="center", va="bottom", fontsize=12)

    plt.savefig(
        f"{results_dir}/{file_name}_sentiment_distribution_{mode}.png"
    )  # Save the plot
    plt.show()  # Show the plot


def plot_predicted_probabilities(Y_probabilities):
    plt.figure(figsize=(10, 6))
    plt.hist(
        Y_probabilities,
        bins=50,
        alpha=0.75,
        color="blue",
        label="Predicted probabilities",
    )
    plt.axvline(
        0.5, color="red", linestyle="dashed", linewidth=1, label="Threshold = 0.5"
    )
    plt.title("Distribution of Predicted Probabilities")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(f"{results_dir}/predicted_probabilities.png")
    plt.show()  # Show the plot
    plt.close()


def plot_predicted_predictions(Y_predictions):
    plt.figure(figsize=(10, 6))
    plt.hist(Y_predictions, bins=50, alpha=0.75, color="blue", label="Predictions")
    plt.axvline(
        0.5, color="red", linestyle="dashed", linewidth=1, label="Threshold = 0.5"
    )
    plt.title("Distribution of Predictions")
    plt.xlabel("Prediction")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(f"{results_dir}/predicted_predictions.png")
    plt.show()  # Show the plot
    plt.close()


def plot_roc_curve(Y_test, Y_probabilities, mode):
    fpr, tpr, thresholds = roc_curve(
        Y_test, Y_probabilities if mode == "bin" else Y_probabilities[:, 1]
    )
    roc_auc = roc_auc_score(
        Y_test, Y_probabilities if mode == "bin" else Y_probabilities[:, 1]
    )
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig(f"{results_dir}/roc_curve.png")
    plt.show()  # Show the plot
    plt.close()
    return roc_auc


def plot_confusion_matrix(Y_test, Y_predictions):
    conf_matrix = confusion_matrix(Y_test, Y_predictions)
    print("Confusion Matrix:\n", conf_matrix)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"{results_dir}/confusion_matrix.png")
    plt.show()  # Show the plot
    plt.close()


def calculate_metrics(Y_test, Y_predictions, Y_probabilities, mode):
    # Calculate and print ROC AUC score
    if mode == "bin":
        roc_auc = roc_auc_score(Y_test, Y_predictions)
    else:
        roc_auc = roc_auc_score(Y_test, Y_predictions, multi_class="ovr")
    print(f"ROC AUC Score: {roc_auc:.2f}")

    if mode == "bin":
        roc_auc_float = roc_auc_score(Y_test, Y_probabilities)
    else:
        roc_auc_float = roc_auc_score(Y_test, Y_probabilities, multi_class="ovr")
    print(f"ROC AUC Score: {roc_auc_float:.2f}")

    # Calculate precision, recall, and F1-score
    precision = precision_score(
        Y_test, Y_predictions, average="binary" if mode == "bin" else "macro"
    )
    recall = recall_score(
        Y_test, Y_predictions, average="binary" if mode == "bin" else "macro"
    )
    f1 = f1_score(Y_test, Y_predictions, average="binary" if mode == "bin" else "macro")

    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")

    # Print classification report
    print("\nClassification Report:\n", classification_report(Y_test, Y_predictions))

    # Summary of evaluation metrics
    metrics_summary = pd.DataFrame(
        {
            "Metric": [
                "Accuracy",
                "Precision",
                "Recall",
                "F1-Score",
                "ROC AUC",
                "Loss",
            ],
            "Value": [accuracy, precision, recall, f1, roc_auc, logloss],
        }
    )
    print("\nSummary of Evaluation Metrics:\n", metrics_summary)
    return roc_auc, roc_auc_float, recall


def save_evaluation_results(
    accuracy, precision, recall, f1, roc_auc, logloss, model_name, test_set_name
):
    results = {
        "Model Name": model_name,
        "Test Set": test_set_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC AUC": roc_auc,
        "Log Loss": logloss,
    }
    results_df = pd.DataFrame([results])
    results_df.to_csv(
        "modelsOnTestSets_results.csv",
        mode="a",
        header=False,
        index=False,
    )
    print("\nResults saved to CSV file.")


# Define the command-line arguments
parser = argparse.ArgumentParser(description="Test a sentiment analysis model.")
parser.add_argument(
    "--mode",
    type=str,
    required=True,
    choices=["bin", "nonbin"],
    help="Mode of the model: nonbin or bin",
)
parser.add_argument("--file_name", type=str, required=True, help="Name of the file")
parser.add_argument("--model_name", type=str, required=True, help="Name of the model")
args = parser.parse_args()

# Access the mode arguments
mode = args.mode
file_name = args.file_name
model_name = args.model_name

# Paths
dir_path = f"savedmodel_{mode}/{model_name}_model"
model_path = f"{dir_path}/{model_name}_{mode}.keras"
history_path = f"{dir_path}/{model_name}_{mode}_history.npy"
vectorizer_path = f"{dir_path}/count_vectorizer_{model_name}_{mode}.pkl"

# Create directory for results if it doesn't exist
results_dir = f"{dir_path}/{file_name}_results"
os.makedirs(results_dir, exist_ok=True)

# Load model and vectorizer
model = load_model(model_path)
with open(vectorizer_path, "rb") as f:
    vec = pickle.load(f)

# Load test dataset
dataset_path = f"preprocessed_datasets/{file_name}_{mode}.csv"
df = pd.read_csv(dataset_path)
X_test = df["reviews"].values
Y_test = df["sentiment"].values

# Transform test data
x_test = vec.transform(X_test.astype("U"))

# Adjust labels for non-binary classification
if mode == "nonbin":
    Y_test += 1
    Y_test = to_categorical(Y_test)  # One-hot encode for non-binary classification

# Predictions
Y_probabilities = model.predict(x_test)

if mode == "bin":
    Y_predictions = (Y_probabilities > 0.5).astype(int)
else:
    Y_predictions = np.argmax(Y_probabilities, axis=1)
    Y_test = np.argmax(Y_test, axis=1)

# Evaluate the model
evaluation = model.evaluate(x_test, Y_test, verbose=True)
loss, accuracy = evaluation[0], evaluation[1]
print(f"Loss: {loss}, Accuracy: {accuracy}")

# Calculate metrics
precision = precision_score(
    Y_test, Y_predictions, average="binary" if mode == "bin" else "macro"
)
recall = recall_score(
    Y_test, Y_predictions, average="binary" if mode == "bin" else "macro"
)
f1 = f1_score(Y_test, Y_predictions, average="binary" if mode == "bin" else "macro")
logloss = log_loss(Y_test, Y_probabilities)

# Calculate ROC AUC
if mode == "bin":
    roc_auc = roc_auc_score(
        Y_test, Y_probabilities
    )  # Only need true labels and probabilities
else:
    roc_auc = roc_auc_score(
        Y_test, Y_probabilities, multi_class="ovr"
    )  # For multi-class

# Save evaluation results
save_evaluation_results(
    accuracy, precision, recall, f1, roc_auc, logloss, model_name, file_name
)

# Plot sentiment distribution
plot_sentiment_distribution(Y_test, file_name, mode)

# Plot confusion matrix and ROC curve
plot_confusion_matrix(Y_test, Y_predictions)
roc_auc = plot_roc_curve(Y_test, Y_probabilities, mode)

# Plot additional graphs
plot_predicted_probabilities(Y_probabilities)
plot_predicted_predictions(Y_predictions)
