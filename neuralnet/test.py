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


def plot_sentiment_comparison(Y_test, Y_predictions):
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(Y_test)), Y_test, alpha=0.5, label="Actual")
    plt.scatter(range(len(Y_test)), Y_predictions, alpha=0.5, label="Predicted", c="r")
    plt.title("Actual vs Predicted Sentiment")
    plt.xlabel("Samples")
    plt.ylabel("Sentiment")
    plt.legend()
    plt.show()


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
    plt.title("Distribution of Predicted probabilities")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()


def plot_predicted_predictions(Y_predictions):
    plt.figure(figsize=(10, 6))
    plt.hist(Y_predictions, bins=50, alpha=0.75, color="blue", label="Predictions")
    plt.axvline(
        0.5, color="red", linestyle="dashed", linewidth=1, label="Threshold = 0.5"
    )
    plt.title("Distribution of predictions")
    plt.xlabel("Prediction")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()


def plot_roc_curve(Y_test, Y_predictions, roc_auc, mode):
    fpr, tpr, thresholds = roc_curve(
        Y_test, Y_predictions if mode == "bin" else Y_predictions[:, 1]
    )
    # Determine the optimal threshold (e.g., the one that maximizes the TPR - FPR difference)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print("Optimal Threshold:", optimal_threshold)
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.show()
    return optimal_threshold


# Plot ROC curve
def plot_roc_curve_float(Y_test, Y_probabilities, roc_auc_float, mode):
    fpr, tpr, thresholds = roc_curve(
        Y_test, Y_probabilities if mode == "bin" else Y_probabilities[:, 1]
    )
    # Determine the optimal threshold (e.g., the one that maximizes the TPR - FPR difference)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold_float = thresholds[optimal_idx]
    print("Optimal Float Threshold:", optimal_threshold_float)
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc_float:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Float")
    plt.legend(loc="lower right")
    plt.show()
    return optimal_threshold_float


def plot_confusion_matrix(Y_test, Y_predictions):
    conf_matrix = confusion_matrix(Y_test, Y_predictions)
    print("Confusion Matrix:\n", conf_matrix)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


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

    # Calculate weighted accuracy
    weights = np.abs(Y_probabilities - 0.5) * 2
    if mode == "bin":
        correct_predictions = Y_test.flatten() == Y_predictions.flatten()
    else:
        correct_predictions = Y_test == Y_predictions

    weighted_accuracy = np.sum(correct_predictions * weights.flatten()) / np.sum(
        weights.flatten()
    )
    print(f"Weighted Accuracy: {weighted_accuracy:.2f}")

    # Calculate Brier score
    brier_score = np.mean((Y_probabilities - Y_test) ** 2)
    print(f"Brier Score: {brier_score:.4f}")

    # Calculate log loss
    logloss = log_loss(Y_test, Y_probabilities)
    print(f"Log Loss: {logloss:.4f}")

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

dir_path = f"savedmodel_{mode}"
model_path = f"{dir_path}/savedmodel_{model_name}_{mode}.keras"
model = load_model(model_path)

with open(f"{dir_path}/count_vectorizer_{model_name}_{mode}.pkl", "rb") as f:
    vec = pickle.load(f)

dataset_path = f"preprocessed_datasets/{file_name}_{mode}.csv"
df = pd.read_csv(dataset_path)
X_test = df["reviews"].values
Y_test = df["sentiment"].values

x_test = vec.transform(X_test.astype("U"))

if mode == "nonbin":
    Y_test += 1
    Y_test = to_categorical(Y_test)  # One-hot encode for non-binary classification

# No need for one-hot encoding in binary mode
Y_probabilities = model.predict(x_test)

if mode == "bin":
    Y_predictions = (Y_probabilities > 0.5).astype(int)
else:
    Y_predictions = np.argmax(Y_probabilities, axis=1)
    Y_test = np.argmax(Y_test, axis=1)

history_path = f"savedmodel{mode}/savedmodel{mode}.npy"
history = np.load(history_path, allow_pickle=True).item()

# Plot actual vs predicted sentiment
plot_sentiment_comparison(Y_test, Y_predictions)

# Plot the distribution of predicted Y_probabilities
plot_predicted_probabilities(Y_probabilities)

plot_predicted_predictions(Y_predictions)

# Calculate and print metrics
evaluation = model.evaluate(x_test, Y_test, verbose=True)
loss = evaluation[0]
accuracy = evaluation[1]
print(f"Loss: {loss}, Accuracy: {accuracy}")
roc_auc, roc_auc_float, recall = calculate_metrics(
    Y_test, Y_predictions, Y_probabilities, mode
)

# Plot ROC curve
plot_roc_curve(Y_test, Y_predictions, roc_auc, mode)

plot_roc_curve_float(Y_test, Y_probabilities, roc_auc_float, mode)

# Plot confusion matrix
plot_confusion_matrix(Y_test, Y_predictions)
