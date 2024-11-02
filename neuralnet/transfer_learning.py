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
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# Metrics
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

# Setup paths
mode = args.mode
file_name = args.file_name
model_name = args.model_name
model_dir = f"savedmodel_{mode}/{model_name}_model"
model_path = f"{model_dir}/{model_name}_{mode}.keras"
dir_path = f"savedmodel_{mode}"
new_model_name = f"{model_name}_TL_On_{file_name}_{mode}"
new_model_dir = f"{dir_path}/{new_model_name}_model"
os.makedirs(new_model_dir, exist_ok=True)

# CSV file to save results
results_csv_path = "models_results.csv"

# Load the pretrained model
base_model = tf.keras.models.load_model(model_path)

# Load the original vectorizer used during the pretrained model's training
with open(
    f"{model_dir}/count_vectorizer_{model_name}_{mode}.pkl",
    "rb",
) as f:
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

# Freeze layers except for the last two
for layer in base_model.layers[:-2]:
    layer.trainable = False

# Define the input layer based on the shape of the vectorized data
input_shape = X_hate_speech_vec.shape[1]
input_layer = layers.Input(shape=(input_shape,))  # Unique name here
x = base_model(
    input_layer, training=False
)  # Connect to the base model with frozen layers

# Add fine-tuning layers
x = layers.Dropout(0.5)(x)
new_output = layers.Dense(outp_node, activation="sigmoid")(x)
model = models.Model(
    inputs=input_layer, outputs=new_output, name="transfer_learning_model"
)

# Compile the model with a low learning rate for fine-tuning
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss=loss_func,
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall(),
        tf.keras.metrics.AUC(),
        tf.keras.metrics.MeanSquaredError(),
    ],
)

# Training callbacks
early_stopping = EarlyStopping(
    monitor="val_loss", patience=2, restore_best_weights=True
)
lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.1, patience=2, min_lr=1e-6
)

# Train the model
history = model.fit(
    X_train,
    Y_train,
    validation_data=(X_valid, Y_valid),
    epochs=300,
    batch_size=32,
    callbacks=[early_stopping, lr_schedule],
)

# Evaluate the model on the test set
test_loss, test_accuracy, test_precision, test_recall, test_auc, test_mse = (
    model.evaluate(X_test, Y_test)
)

# Ensure the AUC key exists in history
if "AUC" in history.history:
    history.history["test_auc"] = [test_auc] * len(history.history["AUC"])
else:
    history.history["test_auc"] = [test_auc] * len(
        history.history["accuracy"]
    )  # Use a different metric length

# Save the model and vectorizer
model.save(f"{new_model_dir}/{new_model_name}.keras")
with open(f"{new_model_dir}/count_vectorizer_{new_model_name}.pkl", "wb") as f:
    pickle.dump(vec, f)

# Save history
np.save(f"{new_model_dir}/{new_model_name}_history.npy", history.history)

# Plot metrics
for metric in ["accuracy", "loss", "precision", "recall", "auc", "mean_squared_error"]:
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
}

results_df = pd.DataFrame([results])
results_df.to_csv(
    results_csv_path, mode="a", header=False, index=False
)  # Append results

# Plot model architecture
plot_model(
    model,
    to_file=f"{new_model_dir}/model_plot_{new_model_name}.png",
    show_shapes=True,
    show_layer_names=True,
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
