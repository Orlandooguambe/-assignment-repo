

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Mini-batch iterator
# -----------------------------
class GetMiniBatch:
    """
    Iterator to obtain mini-batches.

    Parameters
    ----------
    X : ndarray, (n_samples, n_features)
    y : ndarray, (n_samples, n_classes)  # one-hot for NN training
    batch_size : int
    seed : int
    """
    def __init__(self, X, y, batch_size=20, seed=0):
        self.batch_size = int(batch_size)
        rng = np.random.default_rng(seed)
        idx = rng.permutation(np.arange(X.shape[0]))
        self._X = X[idx]
        self._y = y[idx]
        self._stop = int(np.ceil(X.shape[0] / self.batch_size))
        self._counter = 0

    def __len__(self):
        return self._stop

    def __getitem__(self, i):
        p0 = i * self.batch_size
        p1 = p0 + self.batch_size
        return self._X[p0:p1], self._y[p0:p1]

    def __iter__(self):
        self._counter = 0
        return self

    def __next__(self):
        if self._counter >= self._stop:
            raise StopIteration
        p0 = self._counter * self.batch_size
        p1 = p0 + self.batch_size
        self._counter += 1
        return self._X[p0:p1], self._y[p0:p1]


# ---------------------------------------
# Scratch Simple 3-layer Neural Net Class
# ---------------------------------------
class ScratchSimpleNeuralNetrowkClassifier:
    """
    Simple three-layer neural network classifier (NumPy only).

    Architecture: input -> Dense(h1) -> act -> Dense(h2) -> act -> Dense(10) -> softmax

    Parameters
    ----------
    n_features : int, default 784
    n_hidden1 : int, default 400
    n_hidden2 : int, default 200
    n_output  : int, default 10
    epochs    : int, default 10
    lr        : float, default 0.1
    batch_size: int, default 20
    activation: str, "tanh" or "sigmoid"
    weight_sigma: float, std for N(0, sigma^2), default 0.01
    seed      : int or None
    verbose   : bool
    """

    def __init__(self,
                 n_features=784,
                 n_hidden1=400,
                 n_hidden2=200,
                 n_output=10,
                 epochs=10,
                 lr=0.1,
                 batch_size=20,
                 activation="tanh",
                 weight_sigma=0.01,
                 seed=0,
                 verbose=True):
        self.n_features = int(n_features)
        self.n_hidden1 = int(n_hidden1)
        self.n_hidden2 = int(n_hidden2)
        self.n_output  = int(n_output)
        self.epochs    = int(epochs)
        self.lr        = float(lr)
        self.batch_size= int(batch_size)
        self.activation= activation
        self.weight_sigma = float(weight_sigma)
        self.seed = seed
        self.verbose = bool(verbose)

        # to be filled during fit
        self.W1 = None; self.b1 = None
        self.W2 = None; self.b2 = None
        self.W3 = None; self.b3 = None

        # histories
        self.train_loss_history = []
        self.val_loss_history   = []
        self.train_acc_history  = []
        self.val_acc_history    = []

    # ---------- initialization ----------
    def _init_weights(self):
        rng = np.random.default_rng(self.seed)
        s = self.weight_sigma
        self.W1 = s * rng.standard_normal((self.n_features, self.n_hidden1)).astype(np.float32)
        self.b1 = np.zeros((self.n_hidden1,), dtype=np.float32)
        self.W2 = s * rng.standard_normal((self.n_hidden1, self.n_hidden2)).astype(np.float32)
        self.b2 = np.zeros((self.n_hidden2,), dtype=np.float32)
        self.W3 = s * rng.standard_normal((self.n_hidden2, self.n_output)).astype(np.float32)
        self.b3 = np.zeros((self.n_output,), dtype=np.float32)

    # ---------- activations ----------
    @staticmethod
    def _sigmoid(A):
        return 1.0 / (1.0 + np.exp(-A))

    @staticmethod
    def _sigmoid_deriv_from_A(A):
        S = 1.0 / (1.0 + np.exp(-A))
        return (1.0 - S) * S

    @staticmethod
    def _tanh(A):
        return np.tanh(A)

    @staticmethod
    def _tanh_deriv_from_A(A):
        T = np.tanh(A)
        return 1.0 - T*T

    @staticmethod
    def _softmax(A):
        # stable softmax: subtract row-wise max
        A = A - np.max(A, axis=1, keepdims=True)
        expA = np.exp(A)
        return expA / np.sum(expA, axis=1, keepdims=True)

    # ---------- forward ----------
    def _forward(self, X):
        # layer 1
        A1 = X @ self.W1 + self.b1
        if self.activation == "tanh":
            Z1 = self._tanh(A1)
        else:
            Z1 = self._sigmoid(A1)
        # layer 2
        A2 = Z1 @ self.W2 + self.b2
        if self.activation == "tanh":
            Z2 = self._tanh(A2)
        else:
            Z2 = self._sigmoid(A2)
        # layer 3 (logits)
        A3 = Z2 @ self.W3 + self.b3
        Z3 = self._softmax(A3)
        cache = (A1, Z1, A2, Z2, A3, Z3)
        return Z3, cache

    # ---------- loss ----------
    @staticmethod
    def _cross_entropy(y_true_onehot, y_proba, eps=1e-7):
        # mean over batch
        # y_true_onehot shape (B, C), y_proba (B, C)
        p = np.clip(y_proba, eps, 1.0)
        return -np.mean(np.sum(y_true_onehot * np.log(p), axis=1))

    @staticmethod
    def _accuracy(y_true_labels, y_pred_labels):
        return float(np.mean(y_true_labels == y_pred_labels))

    # ---------- backward ----------
    def _backward(self, X, y_onehot, cache):
        A1, Z1, A2, Z2, A3, Z3 = cache
        B = X.shape[0]  # batch size

        # dL/dA3 = (Z3 - Y) / B   (mean CE)
        dA3 = (Z3 - y_onehot) / B  # (B, C)

        # dL/dW3 = Z2^T @ dA3
        dW3 = Z2.T @ dA3                       # (h2, C)
        db3 = np.sum(dA3, axis=0)              # (C,)

        # dL/dZ2 = dA3 @ W3^T
        dZ2 = dA3 @ self.W3.T                  # (B, h2)

        # dL/dA2 = dZ2 ⊙ f'(A2)
        if self.activation == "tanh":
            dA2 = dZ2 * self._tanh_deriv_from_A(A2)
        else:
            dA2 = dZ2 * self._sigmoid_deriv_from_A(A2)

        dW2 = Z1.T @ dA2                       # (h1, h2)
        db2 = np.sum(dA2, axis=0)              # (h2,)

        dZ1 = dA2 @ self.W2.T                  # (B, h1)
        if self.activation == "tanh":
            dA1 = dZ1 * self._tanh_deriv_from_A(A1)
        else:
            dA1 = dZ1 * self._sigmoid_deriv_from_A(A1)

        dW1 = X.T @ dA1                         # (D, h1)
        db1 = np.sum(dA1, axis=0)               # (h1,)

        return dW1, db1, dW2, db2, dW3, db3

    # ---------- update ----------
    def _sgd_step(self, grads):
        dW1, db1, dW2, db2, dW3, db3 = grads
        lr = self.lr
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W3 -= lr * dW3
        self.b3 -= lr * db3

    # ---------- API ----------
    def fit(self, X, y, X_val=None, y_val=None):
        """
        Train with mini-batch SGD.

        X : (n_samples, n_features), float32 in [0,1]
        y : (n_samples,) int labels (0..9)
        X_val, y_val: optional validation set
        """
        # init
        self._init_weights()

        # one-hot for training
        y_oh = np.eye(self.n_output, dtype=np.float32)[y.astype(int)]
        if y_val is not None:
            y_val = y_val.astype(int)
            y_val_oh = np.eye(self.n_output, dtype=np.float32)[y_val]

        for ep in range(1, self.epochs + 1):
            # iterate mini-batches
            mb = GetMiniBatch(X, y_oh, batch_size=self.batch_size, seed=ep)
            for Xb, yb in mb:
                Z3, cache = self._forward(Xb)
                grads = self._backward(Xb, yb, cache)
                self._sgd_step(grads)

            # end epoch: compute train/val loss & acc
            yhat_train_proba = self.predict_proba(X)
            yhat_train = np.argmax(yhat_train_proba, axis=1)
            train_loss = self._cross_entropy(y_oh, yhat_train_proba)
            train_acc  = self._accuracy(np.argmax(y_oh, axis=1), yhat_train)

            self.train_loss_history.append(train_loss)
            self.train_acc_history.append(train_acc)

            if X_val is not None and y_val is not None:
                yhat_val_proba = self.predict_proba(X_val)
                yhat_val = np.argmax(yhat_val_proba, axis=1)
                val_loss = self._cross_entropy(y_val_oh, yhat_val_proba)
                val_acc  = self._accuracy(y_val, yhat_val)

                self.val_loss_history.append(val_loss)
                self.val_acc_history.append(val_acc)

                if self.verbose:
                    print(f"Epoch {ep:02d}/{self.epochs} | "
                          f"train CE: {train_loss:.4f} acc: {train_acc:.4f} | "
                          f"val CE: {val_loss:.4f} acc: {val_acc:.4f}")
            else:
                if self.verbose:
                    print(f"Epoch {ep:02d}/{self.epochs} | "
                          f"train CE: {train_loss:.4f} acc: {train_acc:.4f}")

        return self

    def predict_proba(self, X):
        Z3, _ = self._forward(X.astype(np.float32))
        return Z3

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    # helper to plot learning curves
    def plot_learning_curves(self, show=True, save_path=None):
        it = np.arange(1, len(self.train_loss_history) + 1)
        plt.figure(figsize=(10,4))

        plt.subplot(1,2,1)
        plt.plot(it, self.train_loss_history, label="Train CE")
        if self.val_loss_history:
            plt.plot(it, self.val_loss_history, label="Val CE")
        plt.xlabel("Epoch"); plt.ylabel("Cross-Entropy")
        plt.title("Loss"); plt.grid(True); plt.legend()

        plt.subplot(1,2,2)
        plt.plot(it, self.train_acc_history, label="Train Acc")
        if self.val_acc_history:
            plt.plot(it, self.val_acc_history, label="Val Acc")
        plt.xlabel("Epoch"); plt.ylabel("Accuracy")
        plt.title("Accuracy"); plt.grid(True); plt.legend()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()


# -----------------------------
# Misclassification viewer
# -----------------------------
def show_misclassified_grid(y_pred, y_true, X_flat, num=36):
    """
    Show misclassified images (estimated/correct).
    y_pred, y_true: (n_samples,) ints
    X_flat: (n_samples, 784) floats in [0,1] (will be reshaped to 28x28)
    """
    wrong_idx = np.where(y_pred != y_true)[0]
    if wrong_idx.size == 0:
        print("No misclassifications 🎉")
        return
    num = min(num, wrong_idx.size)
    fig = plt.figure(figsize=(6,6))
    fig.subplots_adjust(left=0, right=0.8, bottom=0, top=0.8, hspace=1, wspace=0.5)
    for i in range(num):
        j = wrong_idx[i]
        ax = fig.add_subplot(6, 6, i+1, xticks=[], yticks=[])
        ax.set_title(f"{y_pred[j]} / {y_true[j]}")
        ax.imshow(X_flat[j].reshape(28,28), cmap="gray")
    plt.show()


# -----------------------------
# Demo / Verification on MNIST
# -----------------------------
def main():
    # 1) Load MNIST
    try:
        from keras.datasets import mnist
    except Exception as e:
        raise SystemExit("Keras not available to load MNIST. Install keras/tensorflow.") from e

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # 2) Flatten and scale to [0,1]
    X_train = X_train.reshape(-1, 784).astype(np.float32) / 255.0
    X_test  = X_test.reshape(-1, 784).astype(np.float32) / 255.0

    # 3) Split train into train/val (80/20)
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    print("Shapes:", X_tr.shape, X_val.shape, X_test.shape)

    # 4) Build and train the scratch NN
    model = ScratchSimpleNeuralNetrowkClassifier(
        n_features=784,
        n_hidden1=400,
        n_hidden2=200,
        n_output=10,
        epochs=12,
        lr=0.1,
        batch_size=20,
        activation="tanh",          # or "sigmoid"
        weight_sigma=0.01,
        seed=0,
        verbose=True
    )
    model.fit(X_tr, y_tr, X_val=X_val, y_val=y_val)

    # 5) Evaluate on val and test
    y_val_pred = model.predict(X_val)
    val_acc = np.mean(y_val_pred == y_val)
    print(f"\nValidation accuracy: {val_acc:.4f}")

    y_test_pred = model.predict(X_test)
    test_acc = np.mean(y_test_pred == y_test)
    print(f"Test accuracy:       {test_acc:.4f}")

    # 6) Learning curves
    model.plot_learning_curves(show=True)

    # 7) Show some misclassifications on validation set
    show_misclassified_grid(y_val_pred, y_val, X_val, num=36)


if __name__ == "__main__":
    main()
