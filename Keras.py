
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # menos logs do TF

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models


# ---------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------

def make_onehot(y, n_classes=None):
    """OneHotEncoder compatível com versões novas/antigas do sklearn."""
    y = np.asarray(y).reshape(-1, 1)
    try:
        enc = OneHotEncoder(sparse_output=False)  # sklearn >= 1.2
    except TypeError:
        enc = OneHotEncoder(sparse=False)         # sklearn < 1.2
    y_oh = enc.fit_transform(y)
    if (n_classes is not None) and (y_oh.shape[1] != n_classes):
        # pad ou recorta — normalmente não necessário
        Z = np.zeros((y_oh.shape[0], n_classes), dtype=float)
        Z[:, :y_oh.shape[1]] = y_oh
        y_oh = Z
    return y_oh


def print_header(title):
    print("\n" + "="*len(title))
    print(title)
    print("="*len(title))


# ---------------------------------------------------------------------
# 1) Iris — binário
# ---------------------------------------------------------------------

def run_iris_binary(iris_csv="Iris.csv", epochs=50):
    print_header("IRIS (binário) — versicolor vs virginica")

    df = pd.read_csv(iris_csv)
    df = df[(df["Species"] == "Iris-versicolor") | (df["Species"] == "Iris-virginica")]

    X = df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]].values.astype("float32")
    y = (df["Species"] == "Iris-virginica").astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )

    model = models.Sequential([
        layers.Dense(16, activation="relu", input_shape=(4,)),
        layers.Dense(8, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, validation_split=0.2, epochs=epochs, verbose=0)

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {acc:.3f} | Test loss: {loss:.4f}")


# ---------------------------------------------------------------------
# 2) Iris — 3 classes
# ---------------------------------------------------------------------

def run_iris_3c(iris_csv="Iris.csv", epochs=50):
    print_header("IRIS (3 classes) — versicolor, virginica, setosa")

    df = pd.read_csv(iris_csv)
    X = df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]].values.astype("float32")
    y = df["Species"].values

    y_oh = make_onehot(y)  # shape: (N,3)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_oh, test_size=0.2, random_state=0, stratify=y
    )

    model = models.Sequential([
        layers.Dense(32, activation="relu", input_shape=(4,)),
        layers.Dense(16, activation="relu"),
        layers.Dense(3, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, validation_split=0.2, epochs=epochs, verbose=0)

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {acc:.3f} | Test loss: {loss:.4f}")


# ---------------------------------------------------------------------
# 3) House Prices — regressão
# ---------------------------------------------------------------------

def run_house_prices(train_csv="train.csv", epochs=60):
    print_header("HOUSE PRICES — regressão (GrLivArea, YearBuilt)")

    df = pd.read_csv(train_csv)
    # features simples (pode adicionar mais)
    X = df[["GrLivArea", "YearBuilt"]].values.astype("float32")
    y = df["SalePrice"].values.astype("float32")

    # train/val/test: 64/16/20
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=0
    )

    # escala ajuda MSE/MAE
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    model = models.Sequential([
        layers.Dense(32, activation="relu", input_shape=(X_train.shape[1],)),
        layers.Dense(16, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, verbose=0)

    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test MAE: {mae:.0f} | Test MSE: {loss:.0f}")


# ---------------------------------------------------------------------
# 4) MNIST — 10 classes (imagens 28x28)
# ---------------------------------------------------------------------

def run_mnist(epochs=5):
    print_header("MNIST — 10 classes (MLP simples)")

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = (x_train.astype("float32") / 255.0)
    x_test  = (x_test.astype("float32") / 255.0)

    model = keras.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=epochs, validation_split=0.1, batch_size=64, verbose=1)

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.3f} | Test loss: {test_loss:.4f}")


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # fixa seeds p/ reprodutibilidade básica
    np.random.seed(0)
    tf.random.set_seed(0)

    # Verifica GPU
    gpu_name = tf.config.list_physical_devices("GPU")
    print(f"GPU: {'encontrada' if gpu_name else 'não encontrada'}")

    # Arquivos esperados no diretório atual:
    # - Iris.csv
    # - train.csv (House Prices)
    iris_csv = "Iris.csv"
    house_csv = "train.csv"

    if not Path(iris_csv).exists():
        print(f"[AVISO] Não encontrei {iris_csv}. Baixe do Kaggle/UCI e deixe ao lado do script.")
    if not Path(house_csv).exists():
        print(f"[AVISO] Não encontrei {house_csv}. Baixe do Kaggle e deixe ao lado do script.")

    # Rode o que tiver disponível:
    if Path(iris_csv).exists():
        run_iris_binary(iris_csv, epochs=50)
        run_iris_3c(iris_csv, epochs=50)

    if Path(house_csv).exists():
        run_house_prices(house_csv, epochs=60)

    # MNIST baixa sozinho
    run_mnist(epochs=5)
