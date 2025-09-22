

import numpy as np
import matplotlib.pyplot as plt


# -------------------------
# Helpers
# -------------------------

def conv1d_output_size(N_in, F, padding=0, stride=1):
    """
    Compute output length for 1D valid conv with padding/stride.
    N_out = floor((N_in + 2P - F)/S) + 1
    """
    return (N_in + 2*padding - F) // stride + 1


# -------------------------
# Simple 1D Conv (single channel, batch=1, stride=1, no padding)
# -------------------------

class SimpleConv1d:
    """
    Single-channel 1D convolution (batch size = 1), stride=1, no padding.

    Forward:
        a[i] = sum_{s=0..F-1} x[i+s] * w[s] + b
    Backward (given delta_a):
        d_b = sum_i delta_a[i]
        d_w[s] = sum_i delta_a[i] * x[i+s]
        d_x[j] = sum_s delta_a[j - s] * w[s]   (out-of-range indices -> 0)
    """
    def __init__(self, w, b, lr=0.1):
        """
        w: (F,) weights
        b: scalar bias in a length-1 array (for consistency)
        """
        self.w = w.astype(np.float64)
        self.b = float(b[0]) if isinstance(b, np.ndarray) else float(b)
        self.lr = float(lr)
        # caches
        self.x = None
        self.a = None

    def forward(self, x):
        """
        x: (N_in,)
        returns a: (N_out,) where N_out = N_in - F + 1
        """
        self.x = x.astype(np.float64)
        F = self.w.shape[0]
        N_in = self.x.shape[0]
        N_out = N_in - F + 1
        a = np.empty(N_out, dtype=np.float64)
        for i in range(N_out):
            a[i] = np.dot(self.x[i:i+F], self.w) + self.b
        self.a = a
        return a

    def backward(self, delta_a):
        """
        delta_a: (N_out,) gradient wrt output a
        returns (d_x, d_w, d_b)
        """
        F = self.w.shape[0]
        N_out = delta_a.shape[0]
        N_in = self.x.shape[0]

        # d_b
        d_b = np.sum(delta_a)

        # d_w
        d_w = np.zeros_like(self.w)
        for s in range(F):
            # sum over output positions i
            # uses x[i+s]
            acc = 0.0
            for i in range(N_out):
                acc += delta_a[i] * self.x[i+s]
            d_w[s] = acc

        # d_x
        d_x = np.zeros_like(self.x)
        for j in range(N_in):
            acc = 0.0
            for s in range(F):
                i = j - s  # position in delta_a that touches x[j] via w[s]
                if 0 <= i < N_out:
                    acc += delta_a[i] * self.w[s]
            d_x[j] = acc

        return d_x, d_w, d_b

    def step(self, d_w, d_b):
        self.w -= self.lr * d_w
        self.b -= self.lr * d_b


# -------------------------
# Multi-channel 1D Conv (advanced: channels, padding, stride, mini-batch)
# -------------------------

class Conv1d:
    """
    1D convolution supporting:
      - multiple input/output channels
      - optional padding (zeros)
      - arbitrary stride
      - optional mini-batch (X shape (B, Cin, N_in))
    Weights: W shape (Cout, Cin, F)
    Bias:    b shape (Cout,)
    Forward (valid):
        a[b, k, i] = sum_c sum_s X[b, c, i*S + s_pad] * W[k, c, s] + b[k]
        (with x padded and s_pad indexing into padded signal)
    """
    def __init__(self, Cout, Cin, F, lr=0.1, padding=0, stride=1, seed=None, init="he"):
        self.Cout = int(Cout)
        self.Cin = int(Cin)
        self.F = int(F)
        self.lr = float(lr)
        self.padding = int(padding)
        self.stride = int(stride)

        rng = np.random.default_rng(seed)
        if init == "xavier":
            std = 1.0 / np.sqrt(Cin * F)
        elif init == "simple":
            std = 0.01
        else:  # He (default)
            std = np.sqrt(2.0 / (Cin * F))
        self.W = (std * rng.standard_normal((Cout, Cin, F))).astype(np.float64)
        self.b = np.zeros((Cout,), dtype=np.float64)

        # caches
        self.X = None           # padded input saved
        self.X_unpadded = None  # keep original for shapes
        self.out_idx = None

    def _pad(self, X):
        """
        If X has shape (B, Cin, N), pad along last dim with zeros.
        If X is (Cin, N), treat B=1.
        """
        if X.ndim == 2:
            X = X[None, ...]  # (1, Cin, N)
        B, Cin, N = X.shape
        P = self.padding
        if P == 0:
            return X
        ZL = np.zeros((B, Cin, P), dtype=X.dtype)
        ZR = np.zeros((B, Cin, P), dtype=X.dtype)
        return np.concatenate([ZL, X, ZR], axis=2)

    def forward(self, X):
        """
        X: (B, Cin, N_in) or (Cin, N_in)
        returns A: (B, Cout, N_out)
        """
        X = X.astype(np.float64)
        if X.ndim == 2:
            X = X[None, ...]
        self.X_unpadded = X
        Xp = self._pad(X)  # (B, Cin, N_in + 2P)
        B, Cin, Np = Xp.shape
        assert Cin == self.Cin

        N_out = conv1d_output_size(N_in=Np, F=self.F, padding=0, stride=self.stride)
        A = np.zeros((B, self.Cout, N_out), dtype=np.float64)

        # compute
        for b in range(B):
            for k in range(self.Cout):
                for i in range(N_out):
                    start = i * self.stride
                    end = start + self.F
                    # (Cin, F)
                    window = Xp[b, :, start:end]
                    A[b, k, i] = np.sum(window * self.W[k]) + self.b[k]

        # cache for backward
        self.X = Xp
        return A

    def backward(self, dA):
        """
        dA: (B, Cout, N_out)
        returns dX: same shape as input X (B, Cin, N_in) or (Cin, N_in)
        and gradients dW, db
        """
        Xp = self.X
        B, Cin, Np = Xp.shape
        _, Cout, N_out = dA.shape

        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)
        dXp = np.zeros_like(Xp)

        for b in range(B):
            for k in range(self.Cout):
                # db
                db[k] += np.sum(dA[b, k])
                for i in range(N_out):
                    start = i * self.stride
                    end = start + self.F
                    # grads wrt weights (Cin, F)
                    dW[k] += dA[b, k, i] * Xp[b, :, start:end]
                    # grads wrt input (Cin, F) → add into dXp window
                    dXp[b, :, start:end] += dA[b, k, i] * self.W[k]

        # unpad dXp to original N_in
        P = self.padding
        if P > 0:
            dX = dXp[:, :, P:-P]
        else:
            dX = dXp

        # gradient step
        self.W -= self.lr * dW
        self.b -= self.lr * db

        # return dX in original rank (drop batch if B==1 and input was 2D)
        if self.X_unpadded.ndim == 2:
            return dX[0], dW, db
        return dX, dW, db


# -------------------------
# Classic blocks: FC, ReLU, Softmax-CE
# -------------------------

class FC:
    def __init__(self, n_in, n_out, lr=0.1, init="xavier", seed=None):
        rng = np.random.default_rng(seed)
        if init == "xavier":
            std = 1.0 / np.sqrt(n_in)
        elif init == "he":
            std = np.sqrt(2.0 / n_in)
        else:
            std = 0.01
        self.W = (std * rng.standard_normal((n_in, n_out))).astype(np.float64)
        self.B = np.zeros((n_out,), dtype=np.float64)
        self.lr = float(lr)
        self.X = None
        self.dW = None
        self.dB = None

    def forward(self, X):
        self.X = X.astype(np.float64)
        return self.X @ self.W + self.B

    def backward(self, dA):
        self.dW = self.X.T @ dA
        self.dB = np.sum(dA, axis=0)
        dZ = dA @ self.W.T
        # update
        self.W -= self.lr * self.dW
        self.B -= self.lr * self.dB
        return dZ


class ReLU:
    def __init__(self):
        self.A = None

    def forward(self, A):
        self.A = A
        return np.maximum(0.0, A)

    def backward(self, dZ):
        return dZ * (self.A > 0)


class Softmax:
    def forward(self, A):
        A = A - np.max(A, axis=1, keepdims=True)
        e = np.exp(A)
        return e / np.sum(e, axis=1, keepdims=True)

    def backward(self, P, Y_onehot):
        B = P.shape[0]
        return (P - Y_onehot) / B


def cross_entropy(P, Y_onehot, eps=1e-7):
    P = np.clip(P, eps, 1.0)
    return -np.mean(np.sum(Y_onehot * np.log(P), axis=1))


# -------------------------
# Mini-batch
# -------------------------

class GetMiniBatch:
    def __init__(self, X, y, batch_size=64, seed=0):
        self.batch_size = int(batch_size)
        rng = np.random.default_rng(seed)
        idx = rng.permutation(np.arange(X.shape[0]))
        self._X = X[idx]
        self._y = y[idx]
        self._stop = int(np.ceil(X.shape[0] / self.batch_size))
        self._ctr = 0

    def __len__(self): return self._stop
    def __iter__(self):
        self._ctr = 0
        return self
    def __next__(self):
        if self._ctr >= self._stop:
            raise StopIteration
        p0 = self._ctr * self.batch_size
        p1 = p0 + self.batch_size
        self._ctr += 1
        return self._X[p0:p1], self._y[p0:p1]


# -------------------------
# Scratch 1D CNN Classifier
# -------------------------

class Scratch1dCNNClassifier:
    """
    Minimal 1D CNN:
      Conv1d(Cin=1 -> Cout), ReLU, Flatten, FC -> 10, Softmax-CE
    Treats each image as a 1D signal of length 784 with 1 channel.
    """
    def __init__(self, Cout=8, F=5, padding=0, stride=1,
                 lr=0.05, batch_size=128, epochs=5, seed=0, verbose=True):
        self.Cout = int(Cout)
        self.F = int(F)
        self.padding = int(padding)
        self.stride = int(stride)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.seed = seed
        self.verbose = verbose

        self.conv = Conv1d(Cout=self.Cout, Cin=1, F=self.F, lr=self.lr,
                           padding=self.padding, stride=self.stride,
                           seed=self.seed, init="he")
        self.relu = ReLU()
        # FC will be built after we know conv output length
        self.fc = None
        self.softmax = Softmax()

        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []

    def _build_fc(self, N_in_flat):
        self.fc = FC(n_in=N_in_flat, n_out=10, lr=self.lr, init="xavier", seed=self.seed)

    def fit(self, X, y, X_val=None, y_val=None):
        """
        X: (N, 784) float32 in [0,1]; will be reshaped to (N, Cin=1, 784)
        """
        X = X.astype(np.float64).reshape(-1, 1, X.shape[1])
        y = y.astype(int)
        Y = np.eye(10, dtype=np.float64)[y]

        if X_val is not None:
            X_val = X_val.astype(np.float64).reshape(-1, 1, X_val.shape[1])
            y_val = y_val.astype(int)
            Y_val = np.eye(10, dtype=np.float64)[y_val]

        # figure out conv output length to build FC
        tmpA = self.conv.forward(X[:1])       # (1, Cout, N_out)
        N_out = tmpA.shape[2]
        self._build_fc(self.Cout * N_out)

        for ep in range(1, self.epochs + 1):
            mb = GetMiniBatch(X, Y, batch_size=self.batch_size, seed=self.seed + ep)
            for Xb, Yb in mb:
                # Forward
                A1 = self.conv.forward(Xb)            # (B, Cout, N_out)
                Z1 = self.relu.forward(A1)            # same shape
                Z1_flat = Z1.reshape(Z1.shape[0], -1) # (B, Cout*N_out)
                logits = self.fc.forward(Z1_flat)     # (B, 10)
                P = self.softmax.forward(logits)      # (B, 10)

                # Backward
                dA = self.softmax.backward(P, Yb)     # (B, 10)
                dZ1_flat = self.fc.backward(dA)       # (B, Cout*N_out)
                dZ1 = dZ1_flat.reshape(Z1.shape)      # (B, Cout, N_out)
                dA1 = self.relu.backward(dZ1)         # (B, Cout, N_out)
                self.conv.backward(dA1)               # update inside

            # Metrics each epoch
            P_tr = self.predict_proba(X.reshape(X.shape[0], -1))
            y_tr = np.argmax(P_tr, axis=1)
            self.train_loss.append(cross_entropy(P_tr, Y))
            self.train_acc.append(float(np.mean(y_tr == y)))

            if X_val is not None:
                P_va = self.predict_proba(X_val.reshape(X_val.shape[0], -1))
                y_va = np.argmax(P_va, axis=1)
                self.val_loss.append(cross_entropy(P_va, Y_val))
                self.val_acc.append(float(np.mean(y_va == y_val)))
                if self.verbose:
                    print(f"Epoch {ep:02d}/{self.epochs} | "
                          f"train CE {self.train_loss[-1]:.4f} acc {self.train_acc[-1]:.4f} | "
                          f"val CE {self.val_loss[-1]:.4f} acc {self.val_acc[-1]:.4f}")
            else:
                if self.verbose:
                    print(f"Epoch {ep:02d}/{self.epochs} | "
                          f"train CE {self.train_loss[-1]:.4f} acc {self.train_acc[-1]:.4f}")

    def predict_proba(self, Xflat):
        Xb = Xflat.astype(np.float64).reshape(-1, 1, Xflat.shape[1])
        A1 = self.conv.forward(Xb)
        Z1 = self.relu.forward(A1)
        Z1_flat = Z1.reshape(Z1.shape[0], -1)
        logits = self.fc.forward(Z1_flat)
        return self.softmax.forward(logits)

    def predict(self, Xflat):
        return np.argmax(self.predict_proba(Xflat), axis=1)

    def plot_curves(self, show=True):
        it = np.arange(1, len(self.train_loss) + 1)
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.plot(it, self.train_loss, label="Train CE")
        if self.val_loss:
            plt.plot(it, self.val_loss, label="Val CE")
        plt.xlabel("Epoch"); plt.ylabel("CE"); plt.grid(True); plt.legend(); plt.title("Loss")
        plt.subplot(1,2,2)
        plt.plot(it, self.train_acc, label="Train Acc")
        if self.val_acc:
            plt.plot(it, self.val_acc, label="Val Acc")
        plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.grid(True); plt.legend(); plt.title("Accuracy")
        plt.show()


# -------------------------
# Problem 3: Tiny check for SimpleConv1d
# -------------------------

def check_problem3():
    x = np.array([1, 2, 3, 4])
    w = np.array([3, 5, 7])
    b = np.array([1])
    conv = SimpleConv1d(w=w, b=b, lr=0.0)

    # forward
    a = conv.forward(x)
    print("Forward a:", a)  # expect [35, 50]

    # backward with given delta_a
    delta_a = np.array([10, 20])
    d_x, d_w, d_b = conv.backward(delta_a)
    print("delta_b:", np.array([d_b]))                 # expect [30]
    print("delta_w:", d_w)                             # expect [ 50,  80, 110]
    print("delta_x:", d_x)                             # expect [ 30, 110, 170, 140]


# -------------------------
# MNIST loader (keras or openml)
# -------------------------

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
            # reshape to (N, 28, 28) to be consistent
            X_train = X_train.reshape(-1, 28, 28)
            X_test  = X_test.reshape(-1, 28, 28)
        except Exception as e:
            raise SystemExit("Could not load MNIST via tensorflow.keras or sklearn.") from e

    # flatten to length 784
    X_train = X_train.reshape(-1, 784).astype(np.float64)
    X_test  = X_test.reshape(-1, 784).astype(np.float64)
    if normalize:
        X_train /= 255.0
        X_test  /= 255.0
    y_train = y_train.astype(int)
    y_test  = y_test.astype(int)
    return X_train, y_train, X_test, y_test


# -------------------------
# Demo / Verification
# -------------------------

def main():
    print("== Problem 2: output size example ==")
    print("N_in=4, F=3, P=0, S=1 ->", conv1d_output_size(4, 3, 0, 1))  # expect 2

    print("\n== Problem 3: tiny forward/backward check ==")
    check_problem3()

    print("\n== Problem 8: train Scratch1dCNNClassifier on MNIST (short run) ==")
    from sklearn.model_selection import train_test_split
    X_train, y_train, X_test, y_test = load_mnist_flat(normalize=True)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    # Very small epochs for a quick sanity check
    model = Scratch1dCNNClassifier(
        Cout=8, F=5, padding=0, stride=1,
        lr=0.05, batch_size=256, epochs=5, seed=0, verbose=True
    )
    model.fit(X_tr, y_tr, X_val=X_val, y_val=y_val)

    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    print("\nValidation accuracy:", np.mean(y_val_pred == y_val))
    print("Test accuracy:", np.mean(y_test_pred == y_test))
    model.plot_curves(show=True)


if __name__ == "__main__":
    main()
