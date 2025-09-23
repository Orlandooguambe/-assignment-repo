#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TF baixo nível (estilo 1.x via tf.compat.v1) — quatro exercícios em um só:
- Iris binário (sigmoid + BCE)
- Iris 3 classes (softmax + CE)
- House Prices (regressão, MSE)
- MNIST (10 classes, MLP)

Escolha o experimento mudando EXPERIMENT no topo.
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # menos logs do TF

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import tensorflow as tf

# === Compatibilidade TF2 -> TF1 API ===
tf.compat.v1.disable_eager_execution()

# ==========================
# CONFIG
# ==========================
EXPERIMENT = "iris_3c"   # "iris_binary" | "iris_3c" | "house" | "mnist"
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
EPOCHS = 50
SEED = 0
np.random.seed(SEED)

# ==========================
# Mini-batch Iterator
# ==========================
class GetMiniBatch:
    """
    Iterator de mini-batch (embaralha uma vez e itera)
    """
    def __init__(self, X, y, batch_size=32, seed=0):
        self.batch_size = int(batch_size)
        rng = np.random.default_rng(seed)
        idx = rng.permutation(X.shape[0])
        self.X = X[idx]
        self.y = y[idx]
        self._stop = int(np.ceil(X.shape[0] / self.batch_size))
        self._counter = 0

    def __len__(self):
        return self._stop

    def __iter__(self):
        self._counter = 0
        return self

    def __next__(self):
        if self._counter >= self._stop:
            raise StopIteration()
        p0 = self._counter * self.batch_size
        p1 = p0 + self.batch_size
        self._counter += 1
        return self.X[p0:p1], self.y[p0:p1]

# ==========================
# MODELOS (3 camadas MLP)
# ==========================
def build_mlp_logits(X_ph, n_input, n_hidden1, n_hidden2, n_out, std=0.05):
    """
    Constrói MLP: X -> Dense(h1, ReLU) -> Dense(h2, ReLU) -> Dense(n_out) (logits)
    """
    w1 = tf.compat.v1.Variable(tf.random.truncated_normal([n_input,  n_hidden1], stddev=std, seed=SEED))
    b1 = tf.compat.v1.Variable(tf.zeros([n_hidden1]))
    w2 = tf.compat.v1.Variable(tf.random.truncated_normal([n_hidden1, n_hidden2], stddev=std, seed=SEED+1))
    b2 = tf.compat.v1.Variable(tf.zeros([n_hidden2]))
    w3 = tf.compat.v1.Variable(tf.random.truncated_normal([n_hidden2, n_out],   stddev=std, seed=SEED+2))
    b3 = tf.compat.v1.Variable(tf.zeros([n_out]))

    h1 = tf.nn.relu(tf.matmul(X_ph, w1) + b1)
    h2 = tf.nn.relu(tf.matmul(h1,   w2) + b2)
    logits = tf.matmul(h2, w3) + b3
    return logits

# ==========================
# DATASETS
# ==========================
def load_iris_binary():
    """
    Iris versicolor vs virginica — binário (Y shape: (N,1) com {0,1})
    """
    # Baixe "Iris.csv" (Kaggle) ou use seu arquivo local na mesma pasta
    df = pd.read_csv("Iris.csv")
    df = df[(df["Species"] == "Iris-versicolor") | (df["Species"] == "Iris-virginica")]
    X = df[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]].values.astype(np.float32)
    y = df["Species"].values
    y = np.where(y == "Iris-versicolor", 0, 1).astype(np.int64).reshape(-1,1)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)
    X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, test_size=0.2, random_state=SEED)

    # scaler opcional
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr)
    X_val = sc.transform(X_val)
    X_te  = sc.transform(X_te)
    return (X_tr, y_tr), (X_val, y_val), (X_te, y_te)

def load_iris_3c():
    """
    Iris 3 classes — one-hot (Y shape: (N,3))
    """
    df = pd.read_csv("Iris.csv")
    X = df[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]].values.astype(np.float32)
    y_str = df["Species"].values.reshape(-1,1)
    enc = OneHotEncoder(sparse=False)
    y_oh = enc.fit_transform(y_str).astype(np.float32)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y_oh, test_size=0.2, random_state=SEED, stratify=y_oh)
    X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, test_size=0.2, random_state=SEED, stratify=y_tr)

    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr)
    X_val = sc.transform(X_val)
    X_te  = sc.transform(X_te)
    return (X_tr, y_tr), (X_val, y_val), (X_te, y_te)

def load_house_prices():
    """
    House Prices — regressão (alvo SalePrice). Usamos 2 features básicas por padrão,
    mas você pode expandir.
    Retorno: Y shape (N,1), opção log1p para estabilizar.
    """
    df = pd.read_csv("train.csv")
    # features básicas:
    feats = ["GrLivArea","YearBuilt"]
    X = df[feats].values.astype(np.float32)
    y = df["SalePrice"].values.astype(np.float32).reshape(-1,1)

    # log-transform (comum neste dataset):
    y = np.log1p(y)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)
    X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, test_size=0.2, random_state=SEED)

    scX = StandardScaler()
    X_tr = scX.fit_transform(X_tr)
    X_val = scX.transform(X_val)
    X_te  = scX.transform(X_te)

    return (X_tr, y_tr), (X_val, y_val), (X_te, y_te)

def load_mnist_flat():
    """
    MNIST (keras) — classif. 10 classes, entrada achatada 784
    Retorna Y one-hot (N,10)
    """
    from tensorflow.keras.datasets import mnist
    (X_tr, y_tr), (X_te, y_te) = mnist.load_data()
    X_tr = (X_tr.reshape(-1, 784).astype(np.float32))/255.0
    X_te = (X_te.reshape(-1, 784).astype(np.float32))/255.0

    X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, test_size=0.2, random_state=SEED, stratify=y_tr)

    enc = OneHotEncoder(sparse=False)
    y_tr = enc.fit_transform(y_tr.reshape(-1,1)).astype(np.float32)
    y_val = enc.transform(y_val.reshape(-1,1)).astype(np.float32)
    y_te  = enc.transform(y_te.reshape(-1,1)).astype(np.float32)

    return (X_tr, y_tr), (X_val, y_val), (X_te, y_te)

# ==========================
# TREINO COMUM
# ==========================
def train_loop_classification(X_tr, y_tr, X_val, y_val, X_te, y_te,
                              n_input, n_hidden1, n_hidden2, n_classes,
                              lr=1e-3, batch_size=32, epochs=50):
    """
    Classificação (binária: n_classes=1 + sigmoid/BCE, multi: n_classes>1 + softmax/CE)
    """
    X_ph = tf.compat.v1.placeholder(tf.float32, [None, n_input])
    if n_classes == 1:
        Y_ph = tf.compat.v1.placeholder(tf.float32, [None, 1])
    else:
        Y_ph = tf.compat.v1.placeholder(tf.float32, [None, n_classes])

    logits = build_mlp_logits(X_ph, n_input, n_hidden1, n_hidden2, n_classes)

    if n_classes == 1:
        # binário: sigmoid + BCE
        loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y_ph, logits=logits))
        probs = tf.nn.sigmoid(logits)
        preds = tf.cast(probs > 0.5, tf.float32)            # (N,1)
        correct = tf.equal(preds, Y_ph)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    else:
        # multi: softmax + CE
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y_ph, logits=logits))
        preds = tf.argmax(logits, axis=1)
        true  = tf.argmax(Y_ph, axis=1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(preds, true), tf.float32))

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)
    train_op  = optimizer.minimize(loss_op)
    init = tf.compat.v1.global_variables_initializer()

    get_mb = GetMiniBatch(X_tr, y_tr, batch_size=batch_size, seed=SEED)

    with tf.compat.v1.Session() as sess:
        sess.run(init)
        for ep in range(1, epochs+1):
            total_loss = 0.0
            n_steps = 0
            for Xb, Yb in get_mb:
                sess.run(train_op, feed_dict={X_ph: Xb, Y_ph: Yb})
                loss = sess.run(loss_op, feed_dict={X_ph: Xb, Y_ph: Yb})
                total_loss += loss
                n_steps += 1
            tr_loss = total_loss / max(1, n_steps)
            val_loss, val_acc = sess.run([loss_op, accuracy], feed_dict={X_ph: X_val, Y_ph: y_val})
            print(f"Epoch {ep:02d}/{epochs} | train loss {tr_loss:.4f} | val loss {val_loss:.4f} | val acc {val_acc:.3f}")

        te_acc = sess.run(accuracy, feed_dict={X_ph: X_te, Y_ph: y_te})
        print(f"TEST acc: {te_acc:.3f}")

def train_loop_regression(X_tr, y_tr, X_val, y_val, X_te, y_te,
                          n_input, n_hidden1, n_hidden2,
                          lr=1e-3, batch_size=32, epochs=50):
    """
    Regressão (MSE); imprime MSE, RMSE e MAE. Lembrar que usamos log1p nos alvos.
    """
    X_ph = tf.compat.v1.placeholder(tf.float32, [None, n_input])
    Y_ph = tf.compat.v1.placeholder(tf.float32, [None, 1])

    logits = build_mlp_logits(X_ph, n_input, n_hidden1, n_hidden2, 1)
    loss_op = tf.reduce_mean(tf.square(logits - Y_ph))
    mae = tf.reduce_mean(tf.abs(logits - Y_ph))
    rmse = tf.sqrt(tf.reduce_mean(tf.square(logits - Y_ph)))

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)
    train_op  = optimizer.minimize(loss_op)
    init = tf.compat.v1.global_variables_initializer()

    get_mb = GetMiniBatch(X_tr, y_tr, batch_size=batch_size, seed=SEED)

    with tf.compat.v1.Session() as sess:
        sess.run(init)
        for ep in range(1, epochs+1):
            total_loss = 0.0
            n_steps = 0
            for Xb, Yb in get_mb:
                sess.run(train_op, feed_dict={X_ph: Xb, Y_ph: Yb})
                loss = sess.run(loss_op, feed_dict={X_ph: Xb, Y_ph: Yb})
                total_loss += loss
                n_steps += 1
            tr_loss = total_loss / max(1, n_steps)

            v_loss, v_rmse, v_mae = sess.run([loss_op, rmse, mae], feed_dict={X_ph: X_val, Y_ph: y_val})
            print(f"Epoch {ep:02d}/{epochs} | train MSE {tr_loss:.4f} | val MSE {v_loss:.4f} | val RMSE {v_rmse:.3f} | val MAE {v_mae:.3f}")

        te_mse, te_rmse, te_mae = sess.run([loss_op, rmse, mae], feed_dict={X_ph: X_te, Y_ph: y_te})
        print(f"TEST MSE {te_mse:.4f} | RMSE {te_rmse:.3f} | MAE {te_mae:.3f}")
        # Se quiser métricas no espaço original (sem log1p), aplique expm1 fora do gráfico.

# ==========================
# MAIN
# ==========================
if __name__ == "__main__":
    print("GPU:", tf.test.gpu_device_name() or "(sem GPU visível)")

    if EXPERIMENT == "iris_binary":
        (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = load_iris_binary()
        n_input = X_tr.shape[1]
        train_loop_classification(
            X_tr, y_tr, X_val, y_val, X_te, y_te,
            n_input=n_input, n_hidden1=50, n_hidden2=100, n_classes=1,
            lr=LEARNING_RATE, batch_size=BATCH_SIZE, epochs=EPOCHS
        )

    elif EXPERIMENT == "iris_3c":
        (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = load_iris_3c()
        n_input = X_tr.shape[1]
        train_loop_classification(
            X_tr, y_tr, X_val, y_val, X_te, y_te,
            n_input=n_input, n_hidden1=50, n_hidden2=100, n_classes=3,
            lr=LEARNING_RATE, batch_size=BATCH_SIZE, epochs=EPOCHS
        )

    elif EXPERIMENT == "house":
        (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = load_house_prices()
        n_input = X_tr.shape[1]
        train_loop_regression(
            X_tr, y_tr, X_val, y_val, X_te, y_te,
            n_input=n_input, n_hidden1=64, n_hidden2=64,
            lr=LEARNING_RATE, batch_size=BATCH_SIZE, epochs=EPOCHS
        )

    elif EXPERIMENT == "mnist":
        (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = load_mnist_flat()
        n_input = X_tr.shape[1]  # 784
        train_loop_classification(
            X_tr, y_tr, X_val, y_val, X_te, y_te,
            n_input=n_input, n_hidden1=256, n_hidden2=128, n_classes=10,
            lr=LEARNING_RATE, batch_size=BATCH_SIZE, epochs=EPOCHS
        )

    else:
        raise ValueError(f"EXPERIMENT inválido: {EXPERIMENT}")
