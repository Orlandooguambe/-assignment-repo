#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Scratch Linear Regression (NumPy) — single-file version

Includes:
- ScratchLinearRegression class (hypothesis, gradient, prediction, losses)
- MSE function
- Usage examples (synthetic data), learning curve
- No-bias test (Problem 8)
- Polynomial features (Problem 9)
- Optional comparison with scikit-learn (Problem 6)
- Brief theory summaries (Problems 10 and 11)

Gradient Descent derivation (Problem 10):
    J(θ) = (1/(2m)) ∑ (θᵀ x^(i) − y^(i))²
    ∂J/∂θ_j = (1/m) ∑ (h_θ(x^(i)) − y^(i)) x_j^(i)
    θ := θ − α (1/m) Xᵀ (Xθ − y)

No local minima (Problem 11):
    J(θ) is a convex quadratic; Hessian H = (1/m) XᵀX ⪰ 0.
    Therefore there is a unique global minimum (if X has linearly independent columns).
"""

import numpy as np
import matplotlib.pyplot as plt


class ScratchLinearRegression:
    """
    Linear Regression from scratch (batch gradient descent)

    Parameters
    ----------
    num_iter : int
        Number of iterations
    lr : float
        Learning rate (alpha)
    no_bias : bool
        If True, DO NOT include a bias term (theta0)
    verbose : bool
        If True, print training progress

    Attributes
    ----------
    coef_ : ndarray, shape (n_features,)  or (n_features+1,) when bias is included
        Learned parameters (bias is coef_[0] if no_bias=False)
    loss : ndarray, shape (num_iter,)
        Training loss J(θ) per iteration
    val_loss : ndarray, shape (num_iter,)
        Validation loss J(θ) per iteration (zeros if no validation data)
    """

    def __init__(self, num_iter=1000, lr=0.01, no_bias=False, verbose=False):
        self.iter = int(num_iter)
        self.lr = float(lr)
        self.no_bias = bool(no_bias)
        self.verbose = bool(verbose)
        self.loss = np.zeros(self.iter)
        self.val_loss = np.zeros(self.iter)
        self.coef_ = None

    # ---------------- [problem1] linear hypothesis ----------------
    def _linear_hypothesis(self, X):
        """
        h_theta(X) = X @ theta   (bias already included if applicable)
        X : (m, d)
        return : (m,)
        """
        return X @ self.coef_

    # ---------------- [problem2] gradient descent ----------------
    def _gradient_descent(self, X, error):
        """
        Apply one GD step:
            theta := theta - lr * (1/m) * X^T (h_theta(X) - y)

        X : (m, d)  (d includes a column of 1s if we use bias)
        error : (m,) = (pred - y)
        """
        m = X.shape[0]
        grad = (X.T @ error) / m
        self.coef_ = self.coef_ - self.lr * grad

    def _add_bias(self, X):
        if self.no_bias:
            return X
        ones = np.ones((X.shape[0], 1), dtype=float)
        return np.hstack([ones, X])

    # ---------------- training + J(θ) logging [problem5] ----------------
    def fit(self, X, y, X_val=None, y_val=None):
        """
        Train with Batch GD.
        Logs J(θ) = (1/(2m)) * sum (h_theta(x_i) - y_i)^2 into self.loss/self.val_loss.

        X : (m, n), y : (m,)
        X_val, y_val optional
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        Xb = self._add_bias(X)
        m, d = Xb.shape
        self.coef_ = np.zeros(d, dtype=float)

        if X_val is not None and y_val is not None:
            X_val = np.asarray(X_val, dtype=float)
            y_val = np.asarray(y_val, dtype=float).ravel()
            Xb_val = self._add_bias(X_val)
        else:
            Xb_val = None

        for t in range(self.iter):
            # forward
            y_pred = self._linear_hypothesis(Xb)
            error = y_pred - y

            # losses (train/val)
            self.loss[t] = (error @ error) / (2.0 * m)
            if Xb_val is not None:
                y_pred_val = Xb_val @ self.coef_
                val_err = y_pred_val - y_val
                mv = Xb_val.shape[0]
                self.val_loss[t] = (val_err @ val_err) / (2.0 * mv)

            # GD step
            self._gradient_descent(Xb, error)

            if self.verbose and (t % max(1, self.iter // 10) == 0 or t == self.iter - 1):
                if Xb_val is None:
                    print(f"iter {t+1:4d}/{self.iter} | J={self.loss[t]:.6f}")
                else:
                    print(f"iter {t+1:4d}/{self.iter} | J={self.loss[t]:.6f} | J_val={self.val_loss[t]:.6f}")

        return self

    # ---------------- [problem3] prediction ----------------
    def predict(self, X):
        X = np.asarray(X, dtype=float)
        Xb = self._add_bias(X)
        return self._linear_hypothesis(Xb)

    # ---------------- [problem7] learning curve ----------------
    def plot_learning_curve(self, show=True, save_path=None):
        it = np.arange(1, self.iter + 1)
        plt.figure()
        plt.plot(it, self.loss, label="Train J(θ)")
        if np.any(self.val_loss > 0):
            plt.plot(it, self.val_loss, label="Val J(θ)")
        plt.xlabel("Iteration")
        plt.ylabel("Loss J(θ)")
        plt.title("Learning curve (GD)")
        plt.legend()
        plt.grid(True)
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()


# ---------------- [problem4] MSE ----------------
def MSE(y_pred, y):
    """
    Mean Squared Error
    y_pred, y : shape (m,)
    returns mse (float)
    """
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    return float(np.mean((y_pred - y) ** 2))


# ---------------- Usage examples ----------------
def _demo_basic(seed=0, show_plot=True):
    """
    Demo: train/validation on synthetic data + learning curve
    """
    rng = np.random.default_rng(seed)
    m, n = 500, 3
    X = rng.normal(size=(m, n))
    true_w = np.array([2.0, -1.0, 0.5])
    y = X @ true_w + 3.0  # true bias
    y += rng.normal(scale=0.3, size=m)  # noise

    # 80/20 split
    idx = np.arange(m)
    rng.shuffle(idx)
    tr = idx[:400]
    va = idx[400:]
    X_tr, y_tr = X[tr], y[tr]
    X_va, y_va = X[va], y[va]

    model = ScratchLinearRegression(num_iter=2000, lr=0.05, no_bias=False, verbose=True)
    model.fit(X_tr, y_tr, X_val=X_va, y_val=y_va)

    print("Learned coefficients (bias is coef_[0]):", model.coef_)
    y_hat = model.predict(X_va)
    print("Validation MSE:", MSE(y_hat, y_va))

    model.plot_learning_curve(show=show_plot)


def _demo_no_bias(seed=0):
    """
    Problem 8: Train WITHOUT bias term (see MSE impact)
    """
    rng = np.random.default_rng(seed)
    m, n = 400, 3
    X = rng.normal(size=(m, n))
    y = X @ np.array([2.0, -1.0, 0.5]) + 3.0
    y += rng.normal(scale=0.3, size=m)

    # split
    idx = np.arange(m)
    rng.shuffle(idx)
    tr = idx[:320]
    va = idx[320:]
    X_tr, y_tr = X[tr], y[tr]
    X_va, y_va = X[va], y[va]

    model_nb = ScratchLinearRegression(num_iter=2000, lr=0.05, no_bias=True, verbose=False)
    model_nb.fit(X_tr, y_tr, X_val=X_va, y_val=y_va)
    print("No-bias coefficients:", model_nb.coef_)
    print("Validation MSE (no bias):", MSE(model_nb.predict(X_va), y_va))


def _demo_poly_features(seed=0):
    """
    Problem 9: Polynomial features (x^2, x^3)
    """
    rng = np.random.default_rng(seed)
    m, n = 500, 3
    X = rng.normal(size=(m, n))
    y = X @ np.array([2.0, -1.0, 0.5]) + 3.0
    y += rng.normal(scale=0.4, size=m)

    # split
    idx = np.arange(m)
    rng.shuffle(idx)
    tr = idx[:400]
    va = idx[400:]
    X_tr, y_tr = X[tr], y[tr]
    X_va, y_va = X[va], y[va]

    # polynomial expansion
    X_poly_tr = np.hstack([X_tr, X_tr**2, X_tr**3])
    X_poly_va = np.hstack([X_va, X_va**2, X_va**3])

    poly_model = ScratchLinearRegression(num_iter=3000, lr=0.01, verbose=False)
    poly_model.fit(X_poly_tr, y_tr, X_val=X_poly_va, y_val=y_va)
    print("Validation MSE (polynomial):", MSE(poly_model.predict(X_poly_va), y_va))


def _demo_compare_sklearn(seed=0):
    """
    Problem 6: Optional comparison with scikit-learn (if installed)
    """
    try:
        from sklearn.linear_model import LinearRegression
    except Exception as e:
        print("scikit-learn not available for comparison:", e)
        return

    rng = np.random.default_rng(seed)
    m, n = 500, 3
    X = rng.normal(size=(m, n))
    y = X @ np.array([2.0, -1.0, 0.5]) + 3.0
    y += rng.normal(scale=0.3, size=m)

    # split
    idx = np.arange(m)
    rng.shuffle(idx)
    tr = idx[:400]
    va = idx[400:]
    X_tr, y_tr = X[tr], y[tr]
    X_va, y_va = X[va], y[va]

    # our model (with internal bias)
    model = ScratchLinearRegression(num_iter=2000, lr=0.05, no_bias=False, verbose=False)
    model.fit(X_tr, y_tr, X_val=X_va, y_val=y_va)
    mse_scratch = MSE(model.predict(X_va), y_va)

    # sklearn: add a column of 1s and disable intercept
    X_tr_b = np.hstack([np.ones((X_tr.shape[0], 1)), X_tr])
    X_va_b = np.hstack([np.ones((X_va.shape[0], 1)), X_va])
    sk = LinearRegression(fit_intercept=False)
    sk.fit(X_tr_b, y_tr)
    mse_sklearn = MSE(sk.predict(X_va_b), y_va)

    print("MSE (scratch):     ", mse_scratch)
    print("MSE (scikit-learn):", mse_sklearn)


if __name__ == "__main__":
    # Toggle these flags as you wish
    RUN_BASIC = True
    RUN_NO_BIAS = True
    RUN_POLY = True
    RUN_COMPARE_SKLEARN = True  # requires scikit-learn

    if RUN_BASIC:
        _demo_basic(seed=0, show_plot=True)

    if RUN_NO_BIAS:
        _demo_no_bias(seed=1)

    if RUN_POLY:
        _demo_poly_features(seed=2)

    if RUN_COMPARE_SKLEARN:
        _demo_compare_sklearn(seed=3)
