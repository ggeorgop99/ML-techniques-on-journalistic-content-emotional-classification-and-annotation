import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import argparse
import os
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# Metrics
def plot_and_save(history, metric, model_dir, new_model_name, mode):
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
    plt.savefig(f"{model_dir}/{new_model_name}_{metric}_{mode}.png")
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
    "--model_path",
    type=str,
    required=True,
    help="Path to the pretrained model",
)
args = parser.parse_args()

# Setup paths
mode = args.mode
file_name = args.file_name
model_path = args.model_path
dir_path = f"savedmodel_{mode}"
new_model_name = f"{model_path}_TL_On_{file_name}_{mode}"
model_dir = f"{dir_path}/{new_model_name}_model"
os.makedirs(model_dir, exist_ok=True)

# Load the pretrained model
base_model = tf.keras.models.load_model(model_path)

# Load the original vectorizer used during the pretrained model's training
with open(
    f"{os.path.dirname(model_path)}/count_vectorizer_{os.path.basename(model_path)}.pkl",
    "rb",
) as f:
    vec = pickle.load(f)

# Load and preprocess the hate speech dataset
dataset_path = f"preprocessed_datasets/{file_name}_{mode}.csv"
hate_speech_df = pd.read_csv(dataset_path)
X_hate_speech = hate_speech_df["text"].values
Y_hate_speech = hate_speech_df["label"].values

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

# Add fine-tuning layers
x = layers.Dropout(0.5)(base_model.layers[-2].output)
new_output = layers.Dense(outp_node, activation="sigmoid")(x)
model = models.Model(inputs=base_model.input, outputs=new_output)

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
    epochs=10,
    batch_size=32,
    callbacks=[early_stopping, lr_schedule],
)

# Evaluate the model on the test set
test_loss, test_accuracy, test_precision, test_recall, test_auc, test_mse = (
    model.evaluate(X_test, Y_test)
)

# Save the model and vectorizer
model.save(f"{model_dir}/{new_model_name}.h5")
model.save(f"{model_dir}/{new_model_name}.keras")
with open(f"{model_dir}/count_vectorizer_{new_model_name}_{mode}.pkl", "wb") as f:
    pickle.dump(vec, f)

# Save history
np.save(f"{model_dir}/{new_model_name}_history.npy", history.history)

# Plot metrics
for metric in ["accuracy", "loss", "precision", "recall", "auc", "mean_squared_error"]:
    plot_and_save(history.history, metric, model_dir, new_model_name, mode)

# Print and save classification report
predictions = model.predict(X_test)
predictions_labels = (
    np.argmax(predictions, axis=1)
    if mode == "nonbin"
    else (predictions > 0.5).astype(int)
)
classification_rep = classification_report(Y_test, predictions_labels, output_dict=True)
report_df = pd.DataFrame(classification_rep).transpose()
report_df.to_csv(f"{model_dir}/{new_model_name}_classification_report.csv", index=False)

print(f"Classification Report:\n{report_df}")
