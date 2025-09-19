#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Scratch SVM Classifier (hard margin, dual ascent-style update)

- Implements the update you specified:
    λ_i ← λ_i + α * ( 1 − Σ_j λ_j y_i y_j K(x_i, x_j) )
  with non-negativity clamp λ_i >= 0.
- Supports linear kernel and polynomial kernel ("polly").
- Tracks support vectors (λ > threshold).
- Predicts using f(x) = Σ_n λ_n y_sv_n K(x, s_n)  and returns sign(f).
- Includes a simple demo on a linearly separable dataset, comparison to sklearn.SVC,
  and a 2D decision-region plot with support vectors highlighted.
"""

import numpy as np
import matplotlib.pyplot as plt


class ScratchSVMClassifier:
    """
    Scratch implementation of SVM classifier (hard margin)

    Parameters
    ----------
    num_iter : int
        Number of passes over the training set
    lr : float
        Learning rate (alpha) for dual updates
    kernel : str
        "linear" or "polly" (polynomial)
    threshold : float
        Threshold to select support vectors (default 1e-5)
    verbose : bool
        If True, print progress

    Polynomial kernel params (only used when kernel="polly")
    ----------
    gamma : float
        Scale of <x, z> term (default 1.0)
    coef0 : float
        Bias term inside the polynomial (default 0.0)
    degree : int
        Degree of the polynomial (default 2)
    """

    def __init__(self, num_iter=200, lr=0.01, kernel='linear', threshold=1e-5,
                 verbose=False, gamma=1.0, coef0=0.0, degree=2):
        # Record hyperparameters
        self.iter = int(num_iter)
        self.lr = float(lr)
        self.kernel = kernel
        self.threshold = float(threshold)
        self.verbose = bool(verbose)

        # Poly params
        self.gamma = float(gamma)
        self.coef0 = float(coef0)
        self.degree = int(degree)

        # Will be set after fit
        self.n_support_vectors = 0
        self.index_support_vectors = None
        self.X_sv = None
        self.lam_sv = None
        self.y_sv = None

        self._gram = None  # cache K(X, X) during training

    # --------- kernels ---------
    def _kernel_matrix(self, A, B):
        """Compute K(A, B) for all pairs (rows of A, rows of B)."""
        if self.kernel == 'linear':
            return A @ B.T
        elif self.kernel == 'polly':
            # (gamma * <x, z> + coef0)^degree
            return (self.gamma * (A @ B.T) + self.coef0) ** self.degree
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")

    def _kernel_vec(self, X, Zrow):
        """Compute vector K(X, z) where z is a single row (shape (d,))."""
        if self.kernel == 'linear':
            return X @ Zrow
        elif self.kernel == 'polly':
            return (self.gamma * (X @ Zrow) + self.coef0) ** self.degree
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")

    # --------- fit ---------
    def fit(self, X, y, X_val=None, y_val=None):
        """
        Learn the SVM classifier via dual-variable updates.
        If validation data is provided, prints accuracy each few iters.

        X : ndarray, shape (n_samples, n_features)
        y : ndarray, shape (n_samples,)
            Labels must be in {-1, +1}. If {0,1} are provided, they’ll be mapped to {-1,+1}.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()

        # Map {0,1} -> {-1,+1} if needed
        unique = np.unique(y)
        if set(unique).issubset({0, 1}):
            y = 2 * y - 1

        if not set(np.unique(y)).issubset({-1, 1}):
            raise ValueError("Labels must be in {-1,+1} (or {0,1} which I will map).")

        n, d = X.shape

        # Precompute Gram matrix K_ij
        self._gram = self._kernel_matrix(X, X)  # (n, n)

        # Initialize lambdas
        lam = np.zeros(n, dtype=float)

        # Iterate
        for t in range(self.iter):
            # One simple scheme: sweep i=0..n-1 and apply the specified update
            # λ_i ← λ_i + α * ( 1 − Σ_j λ_j y_i y_j K_ij )
            # Enforce λ_i >= 0 after each update
            for i in range(n):
                s = np.sum(lam * y * y[i] * self._gram[i, :])
                lam[i] += self.lr * (1.0 - s)
                if lam[i] < 0.0:
                    lam[i] = 0.0

            # Optional progress: compute training accuracy
            if self.verbose and (t % max(1, self.iter // 10) == 0 or t == self.iter - 1):
                yhat = self._predict_with_lam(X, lam, y, X)  # predict on train
                acc = (yhat == y).mean()
                msg = f"iter {t+1:4d}/{self.iter} | train acc={acc:.3f} | #pos(lam)={(lam>self.threshold).sum()}"
                if X_val is not None and y_val is not None:
                    y_val_bin = np.asarray(y_val).ravel()
                    if set(np.unique(y_val_bin)).issubset({0,1}):
                        y_val_bin = 2*y_val_bin - 1
                    yhat_val = self._predict_with_lam(X, lam, y, X_val)
                    acc_val = (yhat_val == y_val_bin).mean()
                    msg += f" | val acc={acc_val:.3f}"
                print(msg)

        # Determine support vectors: lam > threshold
        sv_mask = lam > self.threshold
        self.index_support_vectors = np.where(sv_mask)[0]
        self.n_support_vectors = sv_mask.sum()
        self.X_sv = X[sv_mask]
        self.lam_sv = lam[sv_mask]
        self.y_sv = y[sv_mask]

        return self

    # --------- predict ---------
    def _predict_with_lam(self, X_train, lam, y_train, X_query):
        """
        Internal helper to compute sign( Σ_j lam_j y_j K(x_query, x_j) ).
        """
        K_q = self._kernel_matrix(X_query, X_train)  # shape (m, n)
        f = K_q @ (lam * y_train)                    # shape (m,)
        return np.where(f >= 0.0, 1, -1)

    def predict(self, X):
        """
        Predict labels in {0,1} (maps sign to 0/1).
        Uses only support vectors (as required).
        """
        X = np.asarray(X, dtype=float)
        if self.n_support_vectors == 0:
            raise RuntimeError("Model has no support vectors. Did you call fit()?")

        # f(x) = Σ_n λ_n y_sv_n K(x, s_n)
        K = self._kernel_matrix(X, self.X_sv)        # (m, Nsv)
        f = K @ (self.lam_sv * self.y_sv)            # (m,)
        y_sign = np.where(f >= 0.0, 1, -1)
        return ((y_sign + 1) // 2).astype(int)       # map {-1,+1} -> {0,1}


# =========================
# Demo / Verification code
# =========================

def _make_separable_2d(n_per_class=60, seed=0):
    """
    Generate an easy linearly separable 2D dataset for demo.
    """
    rng = np.random.default_rng(seed)
    mean1, mean2 = np.array([2.0, 2.0]), np.array([-2.0, -2.0])
    cov = np.array([[0.6, 0.0], [0.0, 0.6]])

    X_pos = rng.multivariate_normal(mean1, cov, size=n_per_class)
    X_neg = rng.multivariate_normal(mean2, cov, size=n_per_class)

    X = np.vstack([X_pos, X_neg])
    y = np.hstack([np.ones(n_per_class, dtype=int),
                   np.zeros(n_per_class, dtype=int)])
    return X, y


def _plot_decision_region(model, X, y, title="Decision Region", show_sv=True):
    """
    Plot 2D decision region; highlight support vectors if requested.
    """
    x1_min, x1_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    x2_min, x2_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0

    xx1, xx2 = np.meshgrid(
        np.linspace(x1_min, x1_max, 301),
        np.linspace(x2_min, x2_max, 301)
    )
    grid = np.c_[xx1.ravel(), xx2.ravel()]
    Z = model.predict(grid).reshape(xx1.shape)

    plt.figure()
    plt.contourf(xx1, xx2, Z, alpha=0.25, levels=[-0.1, 0.5, 1.1])
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c='tab:blue', edgecolor='k', label='class 0')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='tab:orange', edgecolor='k', label='class 1')

    if show_sv and getattr(model, "X_sv", None) is not None and model.n_support_vectors > 0:
        plt.scatter(model.X_sv[:, 0], model.X_sv[:, 1],
                    s=120, facecolors='none', edgecolors='red', linewidths=2,
                    label='support vectors')

    plt.legend()
    plt.title(title)
    plt.xlabel("x1"); plt.ylabel("x2")
    plt.grid(True)
    plt.show()


def _demo_compare_sklearn():
    """
    [Problem 4 & 5] Train scratch SVM on a simple dataset,
    compare with sklearn SVC, and visualize decision region.
    """
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    from sklearn.model_selection import train_test_split

    X, y01 = _make_separable_2d(n_per_class=70, seed=3)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y01, test_size=0.3, random_state=42, stratify=y01)

    # ---- Scratch (linear) ----
    svm = ScratchSVMClassifier(num_iter=300, lr=0.01, kernel='linear', threshold=1e-5, verbose=True)
    svm.fit(X_tr, y_tr, X_val=X_te, y_val=y_te)
    y_pred_scratch = svm.predict(X_te)

    acc = accuracy_score(y_te, y_pred_scratch)
    prec = precision_score(y_te, y_pred_scratch, zero_division=0)
    rec = recall_score(y_te, y_pred_scratch, zero_division=0)

    print("\n=== Scratch SVM (linear) ===")
    print(f"Support vectors: {svm.n_support_vectors}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")

    _plot_decision_region(svm, X_te, y_te, title="Scratch SVM — Decision Region")

    # ---- sklearn SVC (hard-ish margin via large C) ----
    svc = SVC(kernel='linear', C=1e6)  # large C approximates hard margin on separable data
    svc.fit(X_tr, y_tr)
    y_pred_svc = svc.predict(X_te)

    acc_s = accuracy_score(y_te, y_pred_svc)
    prec_s = precision_score(y_te, y_pred_svc, zero_division=0)
    rec_s = recall_score(y_te, y_pred_svc, zero_division=0)

    print("\n=== sklearn SVC (linear, C≈∞) ===")
    print(f"Accuracy : {acc_s:.4f}")
    print(f"Precision: {prec_s:.4f}")
    print(f"Recall   : {rec_s:.4f}")

    # Show sklearn decision region for comparison
    class _WrapSk:
        # tiny wrapper so we can reuse the same plotter
        def __init__(self, svc):
            self.svc = svc
            self.X_sv = svc.support_vectors_
            self.n_support_vectors = self.X_sv.shape[0]
        def predict(self, X):
            return self.svc.predict(X)

    _plot_decision_region(_WrapSk(svc), X_te, y_te, title="sklearn SVC — Decision Region")


def _demo_polynomial_kernel():
    """
    [Problem 6] Show polynomial kernel ("polly") on a dataset that is not linearly separable.
    """
    # Circle vs. ring (not linearly separable)
    rng = np.random.default_rng(7)
    n = 80
    angles = rng.uniform(0, 2*np.pi, size=n)
    inner_r = 1.0 + 0.1*rng.standard_normal(size=n)
    outer_r = 2.2 + 0.15*rng.standard_normal(size=n)

    X_inner = np.c_[inner_r*np.cos(angles), inner_r*np.sin(angles)]
    X_outer = np.c_[outer_r*np.cos(angles), outer_r*np.sin(angles)]
    X = np.vstack([X_inner, X_outer])
    y = np.hstack([np.zeros(n, dtype=int), np.ones(n, dtype=int)])

    # Train polynomial SVM
    svm_poly = ScratchSVMClassifier(
        num_iter=600, lr=0.01, kernel='polly', degree=3, gamma=1.0, coef0=1.0,
        threshold=1e-5, verbose=True
    )
    svm_poly.fit(X, y)

    print(f"Polynomial kernel — support vectors: {svm_poly.n_support_vectors}")
    _plot_decision_region(svm_poly, X, y, title="Scratch SVM — Polynomial Kernel (degree=3)")


if __name__ == "__main__":
    # Run the demos
    _demo_compare_sklearn()
    _demo_polynomial_kernel()
