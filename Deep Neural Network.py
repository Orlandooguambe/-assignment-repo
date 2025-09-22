

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------
# Initializers
# ---------------------------

class SimpleInitializer:
    """
    Simple Gaussian initializer N(0, sigma^2)
    """
    def __init__(self, sigma=0.01, seed=None):
        self.sigma = float(sigma)
        self.rng = np.random.default_rng(seed)

    def W(self, n_nodes1, n_nodes2):
        return (self.sigma * self.rng.standard_normal((n_nodes1, n_nodes2))).astype(np.float32)

    def B(self, n_nodes2):
        return np.zeros((n_nodes2,), dtype=np.float32)


class XavierInitializer:
    """
    Xavier/Glorot: std = 1/sqrt(n_in)
    """
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def W(self, n_nodes1, n_nodes2):
        sigma = 1.0 / np.sqrt(n_nodes1)
        return (sigma * self.rng.standard_normal((n_nodes1, n_nodes2))).astype(np.float32)

    def B(self, n_nodes2):
        return np.zeros((n_nodes2,), dtype=np.float32)


class HeInitializer:
    """
    He: std = sqrt(2/n_in)
    """
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def W(self, n_nodes1, n_nodes2):
        sigma = np.sqrt(2.0 / n_nodes1)
        return (sigma * self.rng.standard_normal((n_nodes1, n_nodes2))).astype(np.float32)

    def B(self, n_nodes2):
        return np.zeros((n_nodes2,), dtype=np.float32)


# ---------------------------
# Optimizers
# ---------------------------

class SGD:
    """
    Stochastic Gradient Descent
    """
    def __init__(self, lr=0.1):
        self.lr = float(lr)

    def update(self, layer):
        layer.W -= self.lr * layer.dW
        layer.B -= self.lr * layer.dB
        return layer


class AdaGrad:
    """
    AdaGrad optimizer with per-parameter adapted step sizes
    """
    def __init__(self, lr=0.1, eps=1e-8):
        self.lr = float(lr)
        self.eps = float(eps)

    def update(self, layer):
        # create caches on first use
        if not hasattr(layer, "hW"):
            layer.hW = np.zeros_like(layer.W)
            layer.hB = np.zeros_like(layer.B)
        layer.hW += layer.dW * layer.dW
        layer.hB += layer.dB * layer.dB

        layer.W -= self.lr * (layer.dW / (np.sqrt(layer.hW) + self.eps))
        layer.B -= self.lr * (layer.dB / (np.sqrt(layer.hB) + self.eps))
        return layer


# ---------------------------
# Fully Connected Layer
# ---------------------------

class FC:
    """
    Fully-connected layer from n_nodes1 -> n_nodes2
    """
    def __init__(self, n_nodes1, n_nodes2, initializer, optimizer):
        self.n_in = int(n_nodes1)
        self.n_out = int(n_nodes2)
        self.optimizer = optimizer

        # Initialize parameters
        self.W = initializer.W(self.n_in, self.n_out)
        self.B = initializer.B(self.n_out)

        # caches
        self.X = None
        self.dW = None
        self.dB = None

    def forward(self, X):
        """
        X: (batch, n_in)
        returns A: (batch, n_out)
        """
        self.X = X.astype(np.float32)
        return self.X @ self.W + self.B

    def backward(self, dA):
        """
        dA: (batch, n_out) gradient wrt preactivation
        returns dZ: (batch, n_in) gradient for previous layer
        """
        # gradients wrt parameters
        self.dW = self.X.T @ dA                  # (n_in, n_out)
        self.dB = np.sum(dA, axis=0)            # (n_out,)
        # gradient wrt input
        dZ = dA @ self.W.T                      # (batch, n_in)

        # update parameters
        self.optimizer.update(self)
        return dZ


# ---------------------------
# Activations
# ---------------------------

class Tanh:
    def __init__(self):
        self.A = None

    def forward(self, A):
        self.A = A
        return np.tanh(A)

    def backward(self, dZ):
        # derivative of tanh(A) is 1 - tanh^2(A)
        T = np.tanh(self.A)
        return dZ * (1.0 - T * T)


class Sigmoid:
    def __init__(self):
        self.A = None
        self.S = None

    def forward(self, A):
        self.A = A
        self.S = 1.0 / (1.0 + np.exp(-A))
        return self.S

    def backward(self, dZ):
        # d/dA sigmoid(A) = S*(1-S)
        return dZ * self.S * (1.0 - self.S)


class ReLU:
    def __init__(self):
        self.A = None

    def forward(self, A):
        self.A = A
        return np.maximum(0.0, A)

    def backward(self, dZ):
        mask = (self.A > 0).astype(np.float32)
        return dZ * mask


class Softmax:
    """
    Softmax with a convenience CE backward:
    backward(Z, Y_onehot) -> dA = (Z - Y) / batch
    """
    def forward(self, A):
        # stable softmax
        A = A - np.max(A, axis=1, keepdims=True)
        expA = np.exp(A)
        Z = expA / np.sum(expA, axis=1, keepdims=True)
        return Z

    def backward(self, Z, Y_onehot):
        # mean CE loss grad wrt logits (pre-softmax) is (Z - Y) / batch
        B = Z.shape[0]
        return (Z - Y_onehot) / B


# ---------------------------
# Mini-batch iterator
# ---------------------------

class GetMiniBatch:
    def __init__(self, X, y, batch_size=64, seed=0):
        self.batch_size = int(batch_size)
        rng = np.random.default_rng(seed)
        idx = rng.permutation(np.arange(X.shape[0]))
        self._X = X[idx]
        self._y = y[idx]
        self._stop = int(np.ceil(X.shape[0] / self.batch_size))
        self._counter = 0

    def __len__(self):
        return self._stop

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


# ---------------------------
# Deep NN Classifier (modular)
# ---------------------------

class ScratchDeepNeuralNetrowkClassifier:
    """
    Modular deep NN classifier (fully-connected) with arbitrary depth.

    Configure via:
      - hidden_layers: list of (units, activation_name)
      - initializer: 'simple'|'xavier'|'he'
      - optimizer: 'sgd'|'adagrad'
    """
    def __init__(self,
                 n_features=784,
                 n_output=10,
                 hidden_layers=[(400, "tanh"), (200, "tanh")],
                 initializer="xavier",
                 weight_sigma=0.01,
                 optimizer="sgd",
                 lr=0.1,
                 batch_size=64,
                 epochs=10,
                 seed=0,
                 verbose=True):
        self.n_features = int(n_features)
        self.n_output = int(n_output)
        self.hidden_layers = hidden_layers
        self.initializer_name = initializer
        self.weight_sigma = float(weight_sigma)
        self.optimizer_name = optimizer
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.seed = seed
        self.verbose = bool(verbose)

        # built components
        self.layers = []   # [ (FC, act), ..., (FC, Softmax) ]
        self.softmax = Softmax()

        # logs
        self.train_loss_history = []
        self.val_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

    # --- helpers ---
    def _make_initializer(self):
        if self.initializer_name.lower() in ("simple", "gaussian"):
            return SimpleInitializer(self.weight_sigma, seed=self.seed)
        if self.initializer_name.lower() in ("xavier", "glorot"):
            return XavierInitializer(seed=self.seed)
        if self.initializer_name.lower() in ("he", "kaiming"):
            return HeInitializer(seed=self.seed)
        raise ValueError("Unknown initializer")

    def _make_optimizer(self):
        if self.optimizer_name.lower() == "sgd":
            return SGD(lr=self.lr)
        if self.optimizer_name.lower() == "adagrad":
            return AdaGrad(lr=self.lr)
        raise ValueError("Unknown optimizer")

    def _make_activation(self, name):
        n = name.lower()
        if n == "tanh":
            return Tanh()
        if n == "sigmoid":
            return Sigmoid()
        if n == "relu":
            return ReLU()
        if n == "softmax":
            # not used here as a plain activation; we use Softmax + CE backward
            return Softmax()
        raise ValueError(f"Unknown activation: {name}")

    def _build(self):
        self.layers = []
        init = self._make_initializer()
        opt = self._make_optimizer()

        dims = [self.n_features] + [u for (u, _) in self.hidden_layers] + [self.n_output]
        acts = [a for (_, a) in self.hidden_layers]

        # hidden blocks
        for i, act_name in enumerate(acts):
            fc = FC(dims[i], dims[i+1], init, opt)
            act = self._make_activation(act_name)
            self.layers.append((fc, act))

        # output FC (to logits); softmax handled separately
        fc_out = FC(dims[-2], dims[-1], init, opt)
        self.layers.append((fc_out, None))  # last has no hidden activation; softmax separate

    # --- training / inference ---
    @staticmethod
    def _softmax_ce_loss(Z, Y_onehot, eps=1e-7):
        P = np.clip(Z, eps, 1.0)
        return -np.mean(np.sum(Y_onehot * np.log(P), axis=1))

    @staticmethod
    def _accuracy(y_true, y_pred):
        return float(np.mean(y_true == y_pred))

    def fit(self, X, y, X_val=None, y_val=None):
        X = X.astype(np.float32)
        y = y.astype(int)

        # build network
        self._build()

        # one-hot for training
        C = self.n_output
        y_oh = np.eye(C, dtype=np.float32)[y]

        if X_val is not None and y_val is not None:
            X_val = X_val.astype(np.float32)
            y_val = y_val.astype(int)
            y_val_oh = np.eye(C, dtype=np.float32)[y_val]

        for ep in range(1, self.epochs + 1):
            mb = GetMiniBatch(X, y_oh, batch_size=self.batch_size, seed=self.seed + ep)

            for Xb, Yb in mb:
                # forward through all FC+activation blocks
                A = Xb
                caches_act = []
                for (fc, act) in self.layers[:-1]:
                    A = fc.forward(A)
                    Z = act.forward(A)
                    caches_act.append((fc, act))
                    A = Z

                # last FC to logits
                fc_out, _ = self.layers[-1]
                logits = fc_out.forward(A)
                Z3 = self.softmax.forward(logits)

                # backward: softmax + CE
                dA = self.softmax.backward(Z3, Yb)

                # through last FC
                dZ = fc_out.backward(dA)

                # through hidden blocks (reverse)
                for (fc, act) in reversed(caches_act):
                    dA_prev = act.backward(dZ)
                    dZ = fc.backward(dA_prev)

            # end epoch: compute metrics
            yhat_tr_proba = self.predict_proba(X)
            yhat_tr = np.argmax(yhat_tr_proba, axis=1)
            train_loss = self._softmax_ce_loss(yhat_tr_proba, y_oh)
            train_acc = self._accuracy(np.argmax(y_oh, axis=1), yhat_tr)

            self.train_loss_history.append(train_loss)
            self.train_acc_history.append(train_acc)

            if X_val is not None and y_val is not None:
                yhat_va_proba = self.predict_proba(X_val)
                yhat_va = np.argmax(yhat_va_proba, axis=1)
                val_loss = self._softmax_ce_loss(yhat_va_proba, y_val_oh)
                val_acc = self._accuracy(y_val, yhat_va)

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
        A = X.astype(np.float32)
        for (fc, act) in self.layers[:-1]:
            A = act.forward(fc.forward(A))
        fc_out, _ = self.layers[-1]
        logits = fc_out.forward(A)
        return Softmax().forward(logits)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def plot_learning_curves(self, show=True, save_path=None):
        it = np.arange(1, len(self.train_loss_history) + 1)
        plt.figure(figsize=(10,4))

        plt.subplot(1,2,1)
        plt.plot(it, self.train_loss_history, label="Train CE")
        if self.val_loss_history:
            plt.plot(it, self.val_loss_history, label="Val CE")
        plt.xlabel("Epoch"); plt.ylabel("Cross-Entropy"); plt.grid(True); plt.legend(); plt.title("Loss")

        plt.subplot(1,2,2)
        plt.plot(it, self.train_acc_history, label="Train Acc")
        if self.val_acc_history:
            plt.plot(it, self.val_acc_history, label="Val Acc")
        plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.grid(True); plt.legend(); plt.title("Accuracy")

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()


# ---------------------------
# MNIST loader (Keras or OpenML fallback)
# ---------------------------

def load_mnist_flat(normalize=True):
    X_train = y_train = X_test = y_test = None
    try:
        from tensorflow.keras.datasets import mnist
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    except Exception:
        try:
            from sklearn.datasets import fetch_openml
            X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
            y = y.astype(int)
            X_train, X_test = X[:60000], X[60000:]
            y_train, y_test = y[:60000], y[60000:]
        except Exception as e:
            raise SystemExit("Could not load MNIST via tensorflow.keras or sklearn.") from e

    if X_train.ndim == 3:  # (N, 28, 28)
        X_train = X_train.reshape(-1, 784)
        X_test  = X_test.reshape(-1, 784)

    X_train = X_train.astype(np.float32)
    X_test  = X_test.astype(np.float32)
    if normalize:
        X_train /= 255.0
        X_test  /= 255.0
    y_train = y_train.astype(int)
    y_test  = y_test.astype(int)
    return X_train, y_train, X_test, y_test


# ---------------------------
# Demo / Verification
# ---------------------------

def main():
    from sklearn.model_selection import train_test_split

    X_train, y_train, X_test, y_test = load_mnist_flat(normalize=True)

    # 80/20 validation split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    print("Shapes:", X_tr.shape, X_val.shape, X_test.shape)

    # Example 1: tanh/tanh, Xavier, SGD
    model = ScratchDeepNeuralNetrowkClassifier(
        n_features=784, n_output=10,
        hidden_layers=[(400, "tanh"), (200, "tanh")],
        initializer="xavier", optimizer="sgd",
        lr=0.1, batch_size=64, epochs=12, seed=0, verbose=True
    )
    model.fit(X_tr, y_tr, X_val=X_val, y_val=y_val)
    print("\nValidation accuracy:", np.mean(model.predict(X_val) == y_val))
    print("Test accuracy:      ", np.mean(model.predict(X_test) == y_test))
    model.plot_learning_curves(show=True)

    # Example 2: ReLU/ReLU, He, AdaGrad
    model2 = ScratchDeepNeuralNetrowkClassifier(
        n_features=784, n_output=10,
        hidden_layers=[(512, "relu"), (256, "relu")],
        initializer="he", optimizer="adagrad",
        lr=0.05, batch_size=128, epochs=12, seed=1, verbose=True
    )
    model2.fit(X_tr, y_tr, X_val=X_val, y_val=y_val)
    print("\n[ReLU/He/AdaGrad] Validation accuracy:", np.mean(model2.predict(X_val) == y_val))
    print("[ReLU/He/AdaGrad] Test accuracy:      ", np.mean(model2.predict(X_test) == y_test))
    model2.plot_learning_curves(show=True)


if __name__ == "__main__":
    main()
