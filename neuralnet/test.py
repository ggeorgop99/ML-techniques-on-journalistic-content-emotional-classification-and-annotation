import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
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


def apply_platt_scaling(Y_test, Y_probabilities):
    lr = LogisticRegression()  # Logistic regression for Platt scaling
    lr.fit(Y_probabilities.reshape(-1, 1), Y_test)
    calibrated_prob = lr.predict_proba(Y_probabilities.reshape(-1, 1))[:, 1]
    return calibrated_prob


def apply_isotonic_regression(Y_test, Y_probabilities):
    iso_reg = IsotonicRegression(out_of_bounds="clip")
    iso_reg.fit(Y_probabilities, Y_test)
    calibrated_prob = iso_reg.predict(Y_probabilities)
    return calibrated_prob


def mc_dropout_predict(model, x, n_samples=100):
    # Enable dropout during inference by setting `training=True`
    predictions = [model(x, training=True) for _ in range(n_samples)]
    predictions = tf.stack(predictions, axis=0)  # Stack predictions for all samples
    mean_pred = tf.reduce_mean(predictions, axis=0)  # Mean across samples
    uncertainty = tf.math.reduce_std(
        predictions, axis=0
    )  # Standard deviation across samples
    return mean_pred, uncertainty


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

    plt.savefig(f"{results_dir}/{file_name}_sentiment_distribution_{mode}.png")
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
    plt.title("Distribution of Predicted Probabilities")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(f"{results_dir}/predicted_probabilities.png")
    plt.show()
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
    plt.show()
    plt.close()


def plot_roc_curve_float(Y_test, Y_probabilities, mode, roc_auc_float):
    fpr, tpr, thresholds = roc_curve(
        Y_test, Y_probabilities if mode == "bin" else Y_probabilities[:, 1]
    )
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print("Optimal Threshold:", optimal_threshold)
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc_float:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig(f"{results_dir}/roc_curve_float.png")
    plt.show()
    plt.close()


def plot_roc_curve(Y_test, Y_predictions, mode, roc_auc):
    fpr, tpr, thresholds = roc_curve(
        Y_test, Y_predictions if mode == "bin" else Y_predictions[:, 1]
    )
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
    plt.savefig(f"{results_dir}/roc_curve.png")
    plt.show()
    plt.close()


def plot_confusion_matrix(Y_test, Y_predictions):
    conf_matrix = confusion_matrix(Y_test, Y_predictions)
    print("Confusion Matrix:\n", conf_matrix)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"{results_dir}/confusion_matrix.png")
    plt.show()
    plt.close()


def calculate_metrics(Y_test, Y_predictions, Y_probabilities, mode):
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
    accuracy = np.mean(Y_predictions == Y_test)
    logloss = log_loss(Y_test, Y_probabilities)
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

    print("\nClassification Report:\n", classification_report(Y_test, Y_predictions))

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
    return accuracy, logloss, roc_auc, roc_auc_float, recall, precision, f1


def save_evaluation_results(
    accuracy,
    precision,
    recall,
    f1,
    roc_auc,
    roc_auc_float,
    logloss,
    model_name,
    test_set_name,
    threshold,
    uncertain_predictions,
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
        "ROC AUC (float)": roc_auc_float,
        "Uncertain Predictions": uncertain_predictions,
        "Uncertainty Threshold": threshold,
        "Calibration Method": calibration_method,
    }
    results_df = pd.DataFrame([results])
    results_df.to_csv(
        "modelsOnTestSets_results.csv",
        mode="a",
        header=False,
        index=False,
    )
    print("\nResults saved to CSV file.")


parser = argparse.ArgumentParser(description="Test a sentiment analysis model.")
parser.add_argument(
    "--mode",
    type=str,
    required=False,
    choices=["bin", "nonbin"],
    default="bin",
    help="Mode of the model: nonbin or bin",
)
parser.add_argument("--file_name", type=str, required=True, help="Name of the file")
parser.add_argument("--model_name", type=str, required=True, help="Name of the model")
parser.add_argument(
    "--calibration_method",
    type=str,
    required=False,
    choices=["platt", "isotonic"],
    default="isotonic",
    help="Calibration method",
)
parser.add_argument(
    "--testing_method",
    type=str,
    required=False,
    choices=["classic", "mc"],
    default="classic",
    help="Testing method",
)

args = parser.parse_args()

mode = args.mode
file_name = args.file_name
model_name = args.model_name
calibration_method = args.calibration_method
testing_method = args.testing_method

dir_path = f"savedmodel_{mode}/{model_name}_model"
model_path = f"{dir_path}/{model_name}_{mode}.keras"
vectorizer_path = f"{dir_path}/count_vectorizer_{model_name}_{mode}.pkl"

# Create directory for results if it doesn't exist
results_dir = f"{dir_path}/{file_name}_results"
os.makedirs(results_dir, exist_ok=True)

model = load_model(model_path)
with open(vectorizer_path, "rb") as f:
    vec = pickle.load(f)

dataset_path = f"preprocessed_datasets/{file_name}_{mode}.csv"
df = pd.read_csv(dataset_path)
X_test = df["reviews"].values
Y_test = df["sentiment"].values

# Transform test data
x_test = vec.transform(X_test.astype("U"))

if mode == "nonbin":
    Y_test += 1
    Y_test = to_categorical(Y_test)  # One-hot encode for non-binary classification


if testing_method == "classic":
    Y_probabilities = model.predict(x_test)
    threshold = "N/A"
    no_of_uncertain_predictions = "N/A"
    if mode == "bin":
        final_predictions = (Y_probabilities > 0.5).astype(int)
    else:
        final_predictions = np.argmax(Y_probabilities, axis=1)
        Y_test = np.argmax(Y_test, axis=1)

elif testing_method == "mc":
    mean_pred, uncertainty = mc_dropout_predict(model, x_test.toarray())
    Y_probabilities = mean_pred
    if mode == "bin":
        final_predictions = (Y_probabilities > 0.5).numpy().astype(int)
    else:
        final_predictions = np.argmax(Y_probabilities.numpy(), axis=1)
        Y_test = np.argmax(Y_test, axis=1)
    threshold = 0.2  # Define a threshold for uncertainty
    uncertain_predictions = np.where(uncertainty > threshold)[0]
    no_of_uncertain_predictions = len(uncertain_predictions)
    print(f"Total uncertain predictions: {len(uncertain_predictions)}")
    print(
        f"Total uncertain predictions ratio: {len(uncertain_predictions)/len(mean_pred)}"
    )


# Apply post-hoc calibration
if calibration_method == "platt":
    Y_calibrated_prob = apply_platt_scaling(Y_test, Y_probabilities)
elif calibration_method == "isotonic":
    Y_calibrated_prob = apply_isotonic_regression(Y_test, Y_probabilities)

# Update predictions with calibrated probabilities
Y_calibrated_predictions = (Y_calibrated_prob > 0.5).astype(int)

# Calculate metrics
accuracy, logloss, roc_auc, roc_auc_float, recall, precision, f1 = calculate_metrics(
    Y_test, Y_calibrated_predictions, Y_calibrated_prob, mode
)

# Save evaluation results
save_evaluation_results(
    accuracy,
    precision,
    recall,
    f1,
    roc_auc,
    roc_auc_float,
    logloss,
    model_name,
    file_name,
    threshold,
    no_of_uncertain_predictions,
)

# Plot sentiment distribution
plot_sentiment_distribution(Y_test, file_name, mode)

# Plot confusion matrix and ROC curve
plot_confusion_matrix(Y_test, Y_calibrated_predictions)
plot_roc_curve_float(Y_test, Y_calibrated_prob, mode, roc_auc_float)
plot_roc_curve(Y_test, Y_calibrated_predictions, mode, roc_auc)

# Plot additional graphs
plot_predicted_probabilities(Y_calibrated_prob)
plot_predicted_predictions(Y_calibrated_predictions)
