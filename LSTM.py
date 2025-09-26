# =========================
# KERAS RNN SPRINT — COMPLETE
# =========================
# Runs SimpleRNN/GRU/LSTM on IMDB + (optional) Reuters,
# and a tiny ConvLSTM2D demo on synthetic video.
# Compares accuracies for SimpleRNN/GRU/LSTM.
# Includes brief class explanations at the end (printout).

import os, math, random, numpy as np, tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
tf.random.set_seed(0); np.random.seed(0); random.seed(0)

print("TensorFlow:", tf.__version__)
print("GPU devices:", tf.config.list_physical_devices('GPU'))

# =========================
# 1) Text classification dataset (IMDB)
# =========================
# We’ll keep vocab/length small for quick runs.
VOCAB_SIZE   = 10000
SEQ_LEN      = 200
EMB_DIM      = 64
BATCH        = 128
EPOCHS       = 3   # small so we can compare all 3 quickly

# Load & preprocess IMDB
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=VOCAB_SIZE)
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=SEQ_LEN)
x_test  = keras.preprocessing.sequence.pad_sequences(x_test,  maxlen=SEQ_LEN)

def build_rnn_text_model(cell_type="lstm", units=64, bidirectional=False):
    """
    Build a small text classification model:
    Embedding -> (Bi)RNN cell -> Dense
    cell_type in {"simplernn","gru","lstm"}
    """
    inputs = keras.Input(shape=(SEQ_LEN,), dtype="int32")
    x = layers.Embedding(VOCAB_SIZE, EMB_DIM, input_length=SEQ_LEN)(inputs)

    cell = {
        "simplernn": layers.SimpleRNN(units, return_sequences=False),
        "gru":       layers.GRU(units, return_sequences=False),
        "lstm":      layers.LSTM(units, return_sequences=False),
    }[cell_type]

    if bidirectional:
        x = layers.Bidirectional(cell)(x)
    else:
        x = cell(x)

    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs, name=f"{'bi-' if bidirectional else ''}{cell_type}")
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model

def train_eval_rnn(cell_type, bidirectional=False):
    model = build_rnn_text_model(cell_type, units=64, bidirectional=bidirectional)
    print("\n=== Training", model.name, "on IMDB ===")
    model.summary()
    h = model.fit(x_train, y_train, validation_split=0.2,
                  epochs=EPOCHS, batch_size=BATCH, verbose=2)
    test_loss, test_acc = model.evaluate(x_test, y_test, batch_size=BATCH, verbose=0)
    print(f"[{model.name}] Test accuracy: {test_acc:.4f}")
    return model.name, test_acc

results = []
for cell in ["simplernn","gru","lstm"]:
    name, acc = train_eval_rnn(cell_type=cell, bidirectional=False)
    results.append((name, acc))

# Optional: also try bidirectional versions (comment in if you want)
# for cell in ["simplernn","gru","lstm"]:
#     name, acc = train_eval_rnn(cell_type=cell, bidirectional=True)
#     results.append((name, acc))

print("\n=== IMDB Accuracy Comparison ===")
for name, acc in results:
    print(f"{name:12s}  acc={acc:.4f}")

# =========================
# 2) (Optional) Reuters multi-class classification
# =========================
RUN_REUTERS = False  # set True to run (adds a couple minutes)
if RUN_REUTERS:
    (x_tr_r, y_tr_r), (x_te_r, y_te_r) = keras.datasets.reuters.load_data(num_words=VOCAB_SIZE)
    x_tr_r = keras.preprocessing.sequence.pad_sequences(x_tr_r, maxlen=SEQ_LEN)
    x_te_r = keras.preprocessing.sequence.pad_sequences(x_te_r, maxlen=SEQ_LEN)
    num_classes = np.max(y_tr_r) + 1
    print("Reuters classes:", num_classes)

    def build_rnn_reuters(cell_type="lstm", units=64):
        inputs = keras.Input(shape=(SEQ_LEN,), dtype="int32")
        x = layers.Embedding(VOCAB_SIZE, EMB_DIM)(inputs)
        rnn = {"simplernn": layers.SimpleRNN(units),
               "gru":       layers.GRU(units),
               "lstm":      layers.LSTM(units)}[cell_type]
        x = rnn(x)
        outputs = layers.Dense(num_classes, activation="softmax")(x)
        model = keras.Model(inputs, outputs, name=f"reuters-{cell_type}")
        model.compile(optimizer="adam",
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        return model

    for cell in ["simplernn","gru","lstm"]:
        m = build_rnn_reuters(cell)
        print("\n=== Training", m.name, "on Reuters ===")
        m.summary()
        m.fit(x_tr_r, y_tr_r, validation_split=0.2,
              epochs=3, batch_size=128, verbose=2)
        loss, acc = m.evaluate(x_te_r, y_te_r, batch_size=128, verbose=0)
        print(f"[{m.name}] Test accuracy: {acc:.4f}")

# =========================
# 3) ConvLSTM2D demo: tiny video prediction (synthetic moving squares)
# =========================
# We’ll train a ConvLSTM to predict the last frame from the first frames.

def make_moving_squares(num_samples=500, timesteps=15, H=32, W=32, square=5):
    X = np.zeros((num_samples, timesteps, H, W, 1), dtype=np.float32)
    Y = np.zeros((num_samples, H, W, 1), dtype=np.float32)  # target = last frame
    rng = np.random.default_rng(0)
    for n in range(num_samples):
        x = rng.integers(0, H - square)
        y = rng.integers(0, W - square)
        vx = rng.choice([-1,1]); vy = rng.choice([-1,1])
        for t in range(timesteps):
            frame = np.zeros((H,W), dtype=np.float32)
            frame[x:x+square, y:y+square] = 1.0
            X[n, t, ..., 0] = frame
            x += vx; y += vy
            if x < 0 or x > H - square: vx *= -1; x = max(0, min(x, H - square))
            if y < 0 or y > W - square: vy *= -1; y = max(0, min(y, W - square))
        Y[n, ..., 0] = X[n, -1, ..., 0]
    return X, Y

TIMESTEPS, H, W = 10, 32, 32
Xv, Yv = make_moving_squares(num_samples=600, timesteps=TIMESTEPS, H=H, W=W, square=5)
Xv_tr, Yv_tr = Xv[:500], Yv[:500]
Xv_te, Yv_te = Xv[500:], Yv[500:]

def build_convlstm(h=H, w=W, t=TIMESTEPS, ch=1, filters=32):
    inputs = keras.Input(shape=(t, h, w, ch))
    # One or more ConvLSTM2D layers; small network for speed
    x = layers.ConvLSTM2D(filters=filters, kernel_size=3, padding="same", return_sequences=True)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(filters=filters, kernel_size=3, padding="same", return_sequences=False)(x)
    x = layers.BatchNormalization()(x)
    outputs = layers.Conv2D(1, kernel_size=1, activation="sigmoid", padding="same")(x)
    model = keras.Model(inputs, outputs, name="convlstm_pred_last_frame")
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

clstm = build_convlstm()
print("\n=== Training ConvLSTM2D on synthetic moving squares ===")
clstm.summary()
clstm.fit(Xv_tr, Yv_tr, validation_split=0.2, epochs=3, batch_size=16, verbose=2)
loss, acc = clstm.evaluate(Xv_te, Yv_te, batch_size=16, verbose=0)
print(f"[ConvLSTM2D] Test metrics — loss: {loss:.4f} acc(pixel): {acc:.4f}")

# Show 3 qualitative predictions
import matplotlib.pyplot as plt
pred = clstm.predict(Xv_te[:3], verbose=0)
plt.figure(figsize=(9,4))
for i in range(3):
    plt.subplot(3,3,3*i+1); plt.imshow(Xv_te[i,-1,...,0], cmap="gray"); plt.title("GT last"); plt.axis("off")
    plt.subplot(3,3,3*i+2); plt.imshow(pred[i,...,0], cmap="gray");    plt.title("Pred");    plt.axis("off")
    diff = np.abs(pred[i,...,0] - Xv_te[i,-1,...,0])
    plt.subplot(3,3,3*i+3); plt.imshow(diff, cmap="magma");            plt.title("Abs diff");plt.axis("off")
plt.tight_layout(); plt.show()

# =========================
# 4) Short explanations for report
# =========================
explanations = r"""
Keras RNN Family — What each class is for
-----------------------------------------
RNN                  : A generic RNN wrapper that can run a user-provided RNN Cell (e.g., SimpleRNNCell, GRUCell, LSTMCell).
SimpleRNN            : A basic (ungated) RNN layer with tanh state update; fast, but struggles with long-term dependencies.
GRU                  : Gated Recurrent Unit; fewer parameters than LSTM, often comparable accuracy, faster to train/infer.
LSTM                 : Long Short-Term Memory; input/forget/output gates + cell state; most robust to long dependencies.
ConvLSTM2D           : An LSTM whose internal operations are 2D convolutions (spatiotemporal modeling for videos, radar).

SimpleRNNCell        : The per-timestep cell used inside SimpleRNN; you rarely use cells directly unless customizing.
GRUCell              : The per-timestep GRU cell.
LSTMCell             : The per-timestep LSTM cell.
StackedRNNCells      : Utility to stack multiple *cells* into a single RNN layer (alternative to using return_sequences=True
                       and adding another RNN on top).
CuDNNGRU / CuDNNLSTM : Legacy GPU-optimized variants (TF 1.x / early 2.x). Modern TF/Keras fuses GPU kernels automatically,
                       so these are deprecated; just use GRU/LSTM (they pick fast GPU kernels when available).
"""

print(explanations)
