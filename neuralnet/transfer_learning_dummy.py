import tensorflow as tf
import pandas as pd
import pickle
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


# Load the pre-trained model
base_model = tf.keras.models.load_model("path_to_pretrained_model")

# Load and preprocess the hate speech dataset
hate_speech_df = pd.read_csv("path_to_hate_speech_dataset.csv")
hate_speech_df["text"] = hate_speech_df["text"].apply(clean_text)
hate_speech_df["text"] = hate_speech_df["text"].apply(preprocess_text)

X_hate_speech = hate_speech_df["text"].values
Y_hate_speech = hate_speech_df["label"].values  # Adjust based on your dataset

# Freeze all layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# OR Freeze the first 6 layers and unfreeze the rest
# for layer in base_model.layers[:6]:
#     layer.trainable = False
# for layer in base_model.layers[6:]:
#     layer.trainable = True

# OR OR Freeze some layers in the base model
# for layer in base_model.layers[:-2]:
#     layer.trainable = False

# # Add new layers for fine-tuning
# input_layer = Input(shape=(X_train.shape[1],))
# x = base_model.layers[0](input_layer)
# for layer in base_model.layers[1:-2]:
#     x = layer(x)
# x = GlobalAveragePooling1D()(x)
# output_layer = Dense(1, activation='sigmoid')(x)
# model = Model(inputs=input_layer, outputs=output_layer)

# Add a new output layer for the hate speech classification task
new_output = layers.Dense(1, activation="sigmoid")(
    base_model.output
)  # For binary classification

# Create the new model
model = models.Model(inputs=base_model.input, outputs=new_output)

# Compile the model with a low learning rate
model.compile(
    optimizer=Adam(learning_rate=1e-5), loss="binary_crossentropy", metrics=["accuracy"]
)

# ??? Define callbacks
early_stopping = EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)

# Train the model on the hate speech dataset
history = model.fit(
    x_hate_speech,
    Y_hate_speech,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
)

# Save the fine-tuned model
model.save("fine_tuned_model.h5")
