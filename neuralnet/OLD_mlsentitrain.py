# may need to do pip install pydot-ng and sudo apt install graphvizfor it to work

import pandas as pd
import numpy as np
import tensorflow
import pickle
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Activation
from keras_visualizer import visualizer
from tensorflow.keras import layers

# from keras.utils import np_utils
# it's deprecated and produces error -> to categorical is the correct substitution of np_utils and is used like to_categorical(y, num_classes=None)
from tensorflow.keras.utils import to_categorical

mode = "nonbin"
mode1 = "train"

if mode1 == "train":

    dataset = "reviewstars" + mode + ".csv"

    # df = pd.read_csv(dataset)
    df = pd.read_csv(dataset, usecols=[0, 1])
    # sort for easier normalisation
    df = df.sort_values(by=["sentiment"], ascending=True)

    print(df.head(20))

    X = df["reviews"].values
    Y = df["sentiment"].values

    # script to normalize sentiment values to be equal
    # pos and neg sentiment are counted and their diff
    # is removed from X in pos reviews

    if mode == "nonbin":
        N = len(df.loc[df["sentiment"] == -1])
        Z = len(df.loc[df["sentiment"] == 0])
        P = len(df.loc[df["sentiment"] == 1])
        print(N, Z, P)
        if N < Z:
            Z = N
        print(N, Z, P)
        length1 = P - Z
        length2 = N - Z
        a = len(X)
        X = X[: (a - length1)]
        Y = Y[: (a - length1)]
        X = X[length2:]
        Y = Y[length2:]
        print(X)
        print(Y)
        outp_node = 3
        losss = "categorical_crossentropy"

    else:

        Z = len(df.loc[df["sentiment"] == 0])
        P = len(df.loc[df["sentiment"] == 1])
        a = len(X)
        b = len(Y)
        if P > Z:
            length = P - Z
            print(length)
            X = X[: (a - length)]
            Y = Y[: (b - length)]
        else:

            length = Z - P
            print(length)
            X = X[(a - length) :]
            Y = Y[(b - length) :]

        losss = "binary_crossentropy"
        outp_node = 1
    ###############################################
    sns.countplot(Y)
    plt.show()

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.3, random_state=42
    )

    vec = CountVectorizer()
    vec.fit(X_train.astype("U"))
    with open("savedmodel" + mode + "/count_vectorizer.pkl", "wb") as f:
        pickle.dump(vec, f)

    x_train = vec.transform(X_train.astype("U"))
    x_test = vec.transform(X_test.astype("U"))
    if mode == "nonbin":
        Y_test = Y_test + 1  # because it does not like -1
        Y_train = Y_train + 1  # because it does not like -1

        Y_train = to_categorical(Y_train)  # 3 classes to categories for keras
        Y_test = to_categorical(Y_test)

    model = Sequential()
    model.add(Dense(16, input_dim=x_train.shape[1], activation="relu"))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(outp_node, activation="sigmoid"))
    model.compile(loss=losss, optimizer="Adam", metrics=["accuracy"])
    model.summary()
    history = model.fit(
        x_train,
        Y_train,
        validation_data=(x_test, Y_test),
        epochs=100,
        verbose=True,
        batch_size=16,
    )

    model.save("savedmodel" + mode + "/savedmodel" + mode + ".h5")
    model.save("savedmodel" + mode + "/savedmodel" + mode + ".keras")
    np.save("savedmodel" + mode + "/savedmodel" + mode + ".npy", history.history)


#################################################################################################################################
#################################################################################################################################
#################################################################################################################################

# plots

if mode1 == "graphs":

    model = tensorflow.keras.models.load_model(
        "savedmodel" + mode + "/savedmodel" + mode + ".keras"
    )
    history = np.load(
        "savedmodel" + mode + "/savedmodel" + mode + ".npy", allow_pickle="TRUE"
    ).item()
    plot_model(
        model,
        to_file="model_plot_" + mode + ".png",
        show_shapes=True,
        show_layer_names=True,
    )

    # list all data in history

    # summarize history for accuracy
    plt.plot(history["accuracy"])
    plt.plot(history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.show()
    # summarize history for loss
    plt.plot(history["loss"])
    plt.plot(history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.show()
