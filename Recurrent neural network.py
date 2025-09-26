#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ScratchSimpleRNNClassifier — minimal RNN from scratch (NumPy only)

Implements:
- SimpleRNN (tanh): forward() over a sequence, optional backward() (BPTT).
- Problem 2: verifies the small-sequence forward result given in the prompt.

Notes
-----
Shapes follow the exercise:
X : (batch_size, n_sequences, n_features)
Wx: (n_features, n_nodes)
Wh: (n_nodes,   n_nodes)
B : (n_nodes,)
h0: (batch_size, n_nodes)  (default zeros)
"""

import numpy as np


def softmax(logits):
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class SimpleRNN:
    """
    Simple RNN cell unrolled across time (tanh activation).

    Parameters
    ----------
    n_features : int
        Number of input features per time step.
    n_nodes : int
        Number of RNN hidden units.
    return_sequences : bool
        If True, returns hidden states for all time steps (N, T, H).
        If False, returns only the last hidden state (N, H).
    seed : int
        RNG seed for reproducible initialization.

    Attributes
    ----------
    Wx : (n_features, n_nodes)
    Wh : (n_nodes, n_nodes)
    B  : (n_nodes,)
    """

    def __init__(self, n_features, n_nodes, return_sequences=False, seed=0, sigma=0.1):
        self.n_features = n_features
        self.n_nodes = n_nodes
        self.return_sequences = return_sequences

        rng = np.random.default_rng(seed)
        # Small Gaussian init (simple initializer)
        self.Wx = sigma * rng.standard_normal((n_features, n_nodes))
        self.Wh = sigma * rng.standard_normal((n_nodes, n_nodes))
        self.B  = np.zeros((n_nodes,), dtype=float)

        # caches for backward
        self.cache = None

    def forward(self, X, h0=None):
        """
        Forward pass through time.

        Parameters
        ----------
        X : ndarray, shape (N, T, F)
            Input sequence batch.
        h0 : ndarray, optional, shape (N, H)
            Initial hidden state; zeros if None.

        Returns
        -------
        H_all : ndarray
            (N, T, H) if return_sequences else (N, H)
        """
        N, T, F = X.shape
        H = self.n_nodes
        assert F == self.n_features

        if h0 is None:
            h_t = np.zeros((N, H), dtype=float)
        else:
            h_t = h0.astype(float)

        # For backward, store per-time intermediates
        A_list = []   # pre-activation a_t
        H_list = []   # hidden/state h_t
        X_list = []   # x_t (for clarity in backprop)

        for t in range(T):
            x_t = X[:, t, :]                       # (N, F)
            a_t = x_t @ self.Wx + h_t @ self.Wh + self.B   # (N, H)
            h_t = np.tanh(a_t)                     # (N, H)

            X_list.append(x_t)
            A_list.append(a_t)
            H_list.append(h_t.copy())

        # Save cache for optional backward
        self.cache = {
            "X_list": X_list,
            "A_list": A_list,
            "H_list": H_list,
        }

        if self.return_sequences:
            # stack along time: (T, N, H) -> (N, T, H)
            return np.stack(H_list, axis=1)
        else:
            # last hidden only (N, H)
            return H_list[-1]

    def backward(self, dL_dh, h0=None, need_dX=False):
        """
        Backward pass (BPTT). Advanced/optional.

        Parameters
        ----------
        dL_dh : ndarray
            Gradient of loss w.r.t hidden states.
            Accepts either (N, H)  -> interpreted as gradient for last timestep only,
                     or  (N, T, H) -> per-timestep gradient (e.g. if loss at each step).
        h0 : ndarray or None
            Initial hidden state used in forward (only needed for shape checks).
        need_dX : bool
            If True, also returns gradient wrt inputs per time step (N, T, F).

        Returns
        -------
        grads : dict
            {
              "dWx": (F, H),
              "dWh": (H, H),
              "dB" : (H,),
              "dX" : (N, T, F)    # only if need_dX=True
            }
        """
        assert self.cache is not None, "Run forward() before backward()."
        X_list = self.cache["X_list"]
        A_list = self.cache["A_list"]
        H_list = self.cache["H_list"]

        N = X_list[0].shape[0]
        T = len(X_list)
        F = self.n_features
        H = self.n_nodes

        # If only last hidden gradient is provided, expand to per-timestep zeros except last.
        if dL_dh.ndim == 2:
            dH_seq = np.zeros((N, T, H), dtype=float)
            dH_seq[:, -1, :] = dL_dh
        else:
            assert dL_dh.shape == (N, T, H)
            dH_seq = dL_dh

        dWx = np.zeros_like(self.Wx)
        dWh = np.zeros_like(self.Wh)
        dB  = np.zeros_like(self.B)

        dX_seq = np.zeros((N, T, F), dtype=float) if need_dX else None

        # Running gradient wrt h_(t-1), initialized to zero
        dL_dh_prev = np.zeros((N, H), dtype=float)

        for t in reversed(range(T)):
            x_t = X_list[t]               # (N, F)
            a_t = A_list[t]               # (N, H)
            h_t = H_list[t]               # (N, H)
            h_prev = H_list[t-1] if t > 0 else (np.zeros((N, H)) if h0 is None else h0)

            # total gradient wrt h_t (from loss and from future time step)
            dL_dh_total = dH_seq[:, t, :] + dL_dh_prev  # (N, H)

            # derivative through tanh: da = dL/dh * (1 - tanh(a)^2)
            dtanh = (1.0 - np.tanh(a_t)**2)            # (N, H)
            dA = dL_dh_total * dtanh                   # (N, H)

            # Accumulate parameter grads (sum over batch)
            dWx += x_t.T @ dA                           # (F, H)
            dWh += h_prev.T @ dA                        # (H, H)
            dB  += dA.sum(axis=0)                       # (H,)

            # Gradients w.r.t inputs and previous hidden
            if need_dX:
                dX_seq[:, t, :] = dA @ self.Wx.T        # (N, F)

            dL_dh_prev = dA @ self.Wh.T                 # (N, H)

        grads = {"dWx": dWx, "dWh": dWh, "dB": dB}
        if need_dX:
            grads["dX"] = dX_seq
        return grads


# ---------------------------
# Problem 2: tiny forward test
# ---------------------------
def problem2_tiny_forward_check():
    # Given arrays (scaled by /100 as in the prompt)
    x = np.array([[[1, 2],
                   [2, 3],
                   [3, 4]]], dtype=float) / 100.0  # (N=1, T=3, F=2)

    w_x = np.array([[1, 3, 5, 7],
                    [3, 5, 7, 8]], dtype=float) / 100.0  # (F=2, H=4)

    w_h = np.array([[1, 3, 5, 7],
                    [2, 4, 6, 8],
                    [3, 5, 7, 8],
                    [4, 6, 8,10]], dtype=float) / 100.0  # (H=4, H=4)

    b = np.array([1, 1, 1, 1], dtype=float)             # (H=4,)

    N, T, F = x.shape
    H = w_x.shape[1]

    rnn = SimpleRNN(n_features=F, n_nodes=H, return_sequences=False, seed=0)
    # Overwrite weights/bias with the exact values from the prompt
    rnn.Wx = w_x.copy()
    rnn.Wh = w_h.copy()
    rnn.B  = b.copy()

    # initial hidden state h0 = zeros
    h_last = rnn.forward(x, h0=np.zeros((N, H)))
    print("Problem 2 — last hidden state h:")
    print(h_last)
    print("\nExpected (approximately):")
    print(np.array([[0.79494228, 0.81839002, 0.83939649, 0.85584174]]))


# ---------------------------
# Optional: tiny classification demo (no training)
# ---------------------------
def demo_softmax_on_last_hidden():
    """
    Just illustrates how to take the last hidden state and produce class probabilities.
    (No training loop — for completeness only.)
    """
    N, T, F, H, C = 2, 4, 3, 5, 3
    rng = np.random.default_rng(0)
    X = rng.standard_normal((N, T, F)) * 0.1

    rnn = SimpleRNN(n_features=F, n_nodes=H, return_sequences=False, seed=1)
    h_last = rnn.forward(X)                    # (N, H)

    # Simple linear head + softmax
    W_out = rng.standard_normal((H, C)) * 0.1
    b_out = np.zeros((C,))
    logits = h_last @ W_out + b_out           # (N, C)
    probs = softmax(logits)
    print("\nDemo — class probabilities (untrained):")
    print(probs)


if __name__ == "__main__":
    np.set_printoptions(precision=8, suppress=True)
    problem2_tiny_forward_check()
    demo_softmax_on_last_hidden()
