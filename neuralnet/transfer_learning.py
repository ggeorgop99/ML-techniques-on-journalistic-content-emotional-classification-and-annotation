import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import argparse
import os
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras import backend as K


# Define F1-score metric
def f1_score(y_true, y_pred):
    y_pred = K.round(y_pred)  # Round predictions to 0 or 1 for binary classification
    tp = K.sum(K.cast(y_true * y_pred, "float"), axis=0)  # True positives
    fp = K.sum(K.cast((1 - y_true) * y_pred, "float"), axis=0)  # False positives
    fn = K.sum(K.cast(y_true * (1 - y_pred), "float"), axis=0)  # False negatives

    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())
    f1 = 2 * precision * recall / (precision + recall + K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)  # Handle division by zero

    return K.mean(f1)


# Function to plot and save metrics
def plot_and_save(history, metric, new_model_dir, new_model_name, mode):
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
    plt.savefig(f"{new_model_dir}/{new_model_name}_{metric}_{mode}.png")
    plt.show()
    plt.close()


# Argument parsing
parser = argparse.ArgumentParser(description="Fine-tune a sentiment analysis model.")
parser.add_argument(
    "--mode",
    type=str,
    required=True,
    choices=["bin", "nonbin"],
    help="Model mode: bin or nonbin",
)
parser.add_argument(
    "--file_name", type=str, required=True, help="Name of the hate speech dataset file"
)
parser.add_argument(
    "--model_name",
    type=str,
    required=True,
    help="Path to the pretrained model",
)
args = parser.parse_args()

# Setup paths and names
mode = args.mode
file_name = args.file_name
model_name = args.model_name
model_dir = f"savedmodel_{mode}/{model_name}_model"
model_path = f"{model_dir}/{model_name}_{mode}.keras"
dir_path = f"savedmodel_{mode}"
new_model_name = f"{model_name}_TL_On_{file_name}_{mode}"
new_model_dir = f"{dir_path}/{new_model_name}_model"

# Create the new model directory
os.makedirs(new_model_dir, exist_ok=True)

# CSV file to save results
results_csv_path = "models_results.csv"

# Load the pretrained model
base_model = tf.keras.models.load_model(model_path)

# Inspect the base model architecture
print("Base Model Summary:")
base_model.summary()

# Freeze layers except for the last four
for layer in base_model.layers[:-4]:
    layer.trainable = False

# Load the original vectorizer
with open(f"{model_dir}/count_vectorizer_{model_name}_{mode}.pkl", "rb") as f:
    vec = pickle.load(f)

# Load and preprocess the hate speech dataset
dataset_path = f"preprocessed_datasets/{file_name}_{mode}.csv"
hate_speech_df = pd.read_csv(dataset_path)
X_hate_speech = hate_speech_df["reviews"].values
Y_hate_speech = hate_speech_df["sentiment"].values

# Transform the hate speech dataset using the original vectorizer
X_hate_speech_vec = vec.transform(X_hate_speech.astype("U"))

# Split into training, validation, and test sets
X_train, X_temp, Y_train, Y_temp = train_test_split(
    X_hate_speech_vec, Y_hate_speech, test_size=0.3, random_state=42
)
X_valid, X_test, Y_valid, Y_test = train_test_split(
    X_temp, Y_temp, test_size=0.5, random_state=42
)

# Modify labels if mode is nonbin
if mode == "nonbin":
    Y_train = to_categorical(Y_train + 1)
    Y_valid = to_categorical(Y_valid + 1)
    Y_test = to_categorical(Y_test + 1)
    outp_node = 3
    loss_func = "categorical_crossentropy"
else:
    outp_node = 1
    loss_func = "binary_crossentropy"

# Define the input layer based on the shape of the vectorized data
input_shape = X_hate_speech_vec.shape[1]
input_layer = layers.Input(shape=(input_shape,), name="transfer_input_layer")

# Pass the input through the base model
x = base_model(input_layer, training=False)

# Add fine-tuning layers with unique names
x = layers.Dropout(0.5, name="transfer_dropout")(x)
new_output = layers.Dense(outp_node, activation="sigmoid", name="transfer_output")(x)

# Define the new model
model = models.Model(
    inputs=input_layer, outputs=new_output, name="transfer_learning_model"
)

# Compile the model with a low learning rate for fine-tuning
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss=loss_func,
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.MeanSquaredError(name="mse"),
        f1_score,
    ],
)

# Define class weights to handle class imbalance
class_weights = {0: 1.0, 1: 2.0}  # Adjust based on dataset class distribution

# Training callbacks
early_stopping = EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
)
lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
)

tensorboard_callback = TensorBoard(log_dir=f"{new_model_dir}/logs")

# Train the model
history = model.fit(
    X_train,
    Y_train,
    validation_data=(X_valid, Y_valid),
    epochs=100,  # Adjust as needed
    batch_size=32,
    class_weight=class_weights,
    callbacks=[early_stopping, lr_schedule, tensorboard_callback],
    verbose=1,
)


# Evaluate the model on the test set
test_loss, test_accuracy, test_precision, test_recall, test_auc, test_mse, test_f1 = (
    model.evaluate(X_test, Y_test)
)

# Ensure the AUC key exists in history
if "auc" in history.history:
    history.history["test_auc"] = [test_auc] * len(history.history["auc"])
else:
    history.history["test_auc"] = [test_auc] * len(history.history["accuracy"])

# Save the model and vectorizer
model.save(f"{new_model_dir}/{new_model_name}.keras")
with open(f"{new_model_dir}/count_vectorizer_{new_model_name}.pkl", "wb") as f:
    pickle.dump(vec, f)

# Sanity checks
print("New Model Summary:")
model.summary()

# Inspect some predictions
sample_predictions = model.predict(X_test[:10])
print("Sample Predictions:", sample_predictions)
print("Sample True Labels:", Y_test[:10])

# Save training history
np.save(f"{new_model_dir}/{new_model_name}_history.npy", history.history)

# Plot metrics
for metric in ["accuracy", "loss", "precision", "recall", "auc", "mse", "f1_score"]:
    plot_and_save(history.history, metric, new_model_dir, new_model_name, mode)

# Save model results to a CSV
results = {
    "Model Name": new_model_name,
    "Loss": test_loss,
    "Accuracy": test_accuracy,
    "Precision": test_precision,
    "Recall": test_recall,
    "AUC": test_auc,
    "MSE": test_mse,
    "F1 Score": test_f1,
}

# Append results to the CSV
results_df = pd.DataFrame([results])
results_df.to_csv(
    results_csv_path, mode="a", header=not os.path.exists(results_csv_path), index=False
)

# Plot model architecture
model_plot_path = f"{new_model_dir}/model_plot_{new_model_name}.png"

# Remove existing plot file if it exists
if os.path.exists(model_plot_path):
    os.remove(model_plot_path)

plot_model(
    model,
    to_file=model_plot_path,
    show_shapes=True,
    show_layer_names=True,
    expand_nested=True,  # Helps visualize the complete architecture
)

# Print and save classification report
predictions = model.predict(X_test)
predictions_labels = (
    np.argmax(predictions, axis=1)
    if mode == "nonbin"
    else (predictions > 0.5).astype(int)
)
classification_rep = classification_report(Y_test, predictions_labels, output_dict=True)
report_df = pd.DataFrame(classification_rep).transpose()
report_df.to_csv(
    f"{new_model_dir}/{new_model_name}_classification_report.csv", index=False
)
print(f"Classification Report:\n{report_df}")
