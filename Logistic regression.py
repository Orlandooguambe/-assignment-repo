#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Scratch Logistic Regression (NumPy) — single-file

Includes:
- ScratchLogisticRegression class:
    * sigmoid hypothesis, batch gradient descent (with L2 regularization)
    * fit() logs train/val loss per iteration
    * predict_proba() returns probabilities
    * predict() returns labels (threshold=0.5 by default)
    * save_weights() / load_weights()
- Iris verification (versicolor vs virginica) with Accuracy/Precision/Recall
- Learning curve plot
- 2D decision region plot using two features (Sepal width, Petal length)
"""

import numpy as np
import matplotlib.pyplot as plt


# ==========================
# Scratch Logistic Regression
# ==========================

class ScratchLogisticRegression:
    """
    Scratch implementation of logistic regression (batch GD)

    Parameters
    ----------
    num_iter : int
        Number of iterations
    lr : float
        Learning rate
    no_bias : bool
        True if NO bias term is included (default False -> include bias)
    lambda_ : float
        L2 regularization strength (λ). Excludes bias from penalty.
    verbose : bool
        True to print learning progress

    Attributes
    ----------
    coef_ : ndarray, shape (n_features,) or (n_features+1,) when bias included
        Model parameters. If bias is included, coef_[0] is the intercept weight.
    loss : ndarray, shape (num_iter,)
        Train loss per iteration (regularized log loss)
    val_loss : ndarray, shape (num_iter,)
        Validation loss per iteration (0 if no val data)
    """

    def __init__(self, num_iter=1000, lr=0.1, no_bias=False, lambda_=0.0, verbose=False):
        self.iter = int(num_iter)
        self.lr = float(lr)
        self.no_bias = bool(no_bias)
        self.lambda_ = float(lambda_)
        self.verbose = bool(verbose)

        self.loss = np.zeros(self.iter)
        self.val_loss = np.zeros(self.iter)

        self.coef_ = None
        self._eps = 1e-12  # for numerical stability in log-loss

    # ---------- utilities ----------
    def _add_bias(self, X):
        if self.no_bias:
            return X
        ones = np.ones((X.shape[0], 1), dtype=float)
        return np.hstack([ones, X])

    @staticmethod
    def _sigmoid(z):
        # numerically stable sigmoid
        # clip z to avoid overflow in exp
        z = np.clip(z, -40, 40)
        return 1.0 / (1.0 + np.exp(-z))

    def _linear(self, Xb):
        # θ^T x
        return Xb @ self.coef_

    def _hypothesis(self, Xb):
        # hθ(x) = sigmoid(θ^T x)
        return self._sigmoid(self._linear(Xb))

    def _log_loss(self, y_prob, y, n_features_eff):
        """
        Regularized log loss:
            J(θ) = (1/m) Σ [ -y log(p) - (1-y) log(1-p) ] + (λ/(2m)) * ||θ_no_bias||^2
        """
        m = y.shape[0]
        p = np.clip(y_prob, self._eps, 1.0 - self._eps)
        data_term = - (y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean()

        if self.no_bias:
            theta_no_bias = self.coef_
        else:
            theta_no_bias = self.coef_[1:]  # exclude bias

        reg_term = (self.lambda_ / (2.0 * m)) * np.dot(theta_no_bias, theta_no_bias)
        return data_term + reg_term

    # ---------- [problem2] gradient descent ----------
    def _gradient_descent(self, Xb, y, y_prob):
        """
        θ := θ - α * [ (1/m) X^T (p - y) + (λ/m) * θ_no_bias ]
        (bias term not regularized)
        """
        m = Xb.shape[0]
        error = (y_prob - y)  # shape (m,)

        grad = (Xb.T @ error) / m  # shape (d,)

        if self.lambda_ > 0:
            if self.no_bias:
                reg = (self.lambda_ / m) * self.coef_
                grad += reg
            else:
                reg = np.zeros_like(self.coef_)
                reg[1:] = (self.lambda_ / m) * self.coef_[1:]
                grad += reg

        self.coef_ -= self.lr * grad

    # ---------- fit/predict ----------
    def fit(self, X, y, X_val=None, y_val=None):
        """
        Learn logistic regression with batch GD.
        Logs regularized log-loss to self.loss (train) and self.val_loss (val) each iteration.
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
            p = self._hypothesis(Xb)
            # loss
            self.loss[t] = self._log_loss(p, y, d)

            # validation loss
            if Xb_val is not None:
                p_val = self._hypothesis(Xb_val)
                self.val_loss[t] = self._log_loss(p_val, y_val, Xb_val.shape[1])

            # step
            self._gradient_descent(Xb, y, p)

            if self.verbose and (t % max(1, self.iter // 10) == 0 or t == self.iter - 1):
                if Xb_val is None:
                    print(f"iter {t+1:4d}/{self.iter} | J={self.loss[t]:.6f}")
                else:
                    print(f"iter {t+1:4d}/{self.iter} | J={self.loss[t]:.6f} | J_val={self.val_loss[t]:.6f}")

        return self

    def predict_proba(self, X):
        """
        Return probabilities P(y=1|x).
        """
        X = np.asarray(X, dtype=float)
        Xb = self._add_bias(X)
        return self._hypothesis(Xb)

    def predict(self, X, threshold=0.5):
        """
        Return hard labels (0/1) by thresholding predict_proba.
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    # ---------- (Problem 8) save/load ----------
    def save_weights(self, path):
        np.savez(path, coef=self.coef_, iter=self.iter, lr=self.lr,
                 no_bias=self.no_bias, lambda_=self.lambda_)

    def load_weights(self, path):
        data = np.load(path, allow_pickle=False)
        self.coef_ = data["coef"]
        self.iter = int(data["iter"])
        self.lr = float(data["lr"])
        self.no_bias = bool(data["no_bias"])
        self.lambda_ = float(data["lambda_"])


# ======================
# Verification on IRIS
# ======================

def _iris_binary_versicolor_virginica():
    """
    [Problem 5] Train & evaluate on Iris (versicolor vs virginica).
    - Use only two features for decision-region plotting:
        sepal width (index 1) and petal length (index 2).
    - Compute accuracy, precision, recall (scikit-learn).
    - Plot learning curve [Problem 6].
    - Plot decision region [Problem 7].
    """
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score

    iris = load_iris()
    X_all = iris.data  # (150, 4)
    y_all = iris.target  # labels: 0=setosa, 1=versicolor, 2=virginica
    target_names = iris.target_names

    # Filter versicolor (1) and virginica (2)
    mask = (y_all != 0)
    X = X_all[mask]
    y = y_all[mask]
    # Re-label to {0,1}: versicolor->0, virginica->1
    y = (y == 2).astype(int)

    # Two features for 2D plotting: sepal width (1), petal length (2)
    feat_idx = [1, 2]
    X2 = X[:, feat_idx]

    X_train, X_test, y_train, y_test = train_test_split(
        X2, y, test_size=0.3, random_state=42, stratify=y
    )

    # Train our scratch model
    model = ScratchLogisticRegression(
        num_iter=2000, lr=0.1, no_bias=False, lambda_=0.01, verbose=False
    )
    model.fit(X_train, y_train, X_val=X_test, y_val=y_test)

    # Predict
    y_proba = model.predict_proba(X_test)
    y_pred = (y_proba >= 0.5).astype(int)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)

    print("=== Scratch Logistic Regression (Iris: versicolor vs virginica) ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")

    # Learning curve
    plt.figure()
    plt.plot(model.loss, label="Train loss (log-loss)")
    if np.any(model.val_loss > 0):
        plt.plot(model.val_loss, label="Val loss (log-loss)")
    plt.xlabel("Iteration")
    plt.ylabel("Loss J(θ)")
    plt.title("Learning Curve — Logistic Regression")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Decision region (2D)
    _plot_decision_region_2d(model, X_train, y_train, X_test, y_test,
                             feat_names=[iris.feature_names[i] for i in feat_idx])

    # Optional: compare with scikit-learn LogisticRegression
    try:
        from sklearn.linear_model import LogisticRegression
        sk = LogisticRegression(penalty="l2", C=1.0/(2*model.lambda_ + 1e-12),
                                solver="lbfgs")
        sk.fit(X_train, y_train)
        y_sk = sk.predict(X_test)
        acc_sk = accuracy_score(y_test, y_sk)
        print(f"scikit-learn LogisticRegression Accuracy: {acc_sk:.4f}")
    except Exception as e:
        print("scikit-learn comparison skipped:", e)


def _plot_decision_region_2d(model, X_train, y_train, X_test, y_test, feat_names=None):
    """
    Plot decision region for 2D input using a trained ScratchLogisticRegression.
    """
    x1_min, x1_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
    x2_min, x2_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5

    xx1, xx2 = np.meshgrid(
        np.linspace(x1_min, x1_max, 300),
        np.linspace(x2_min, x2_max, 300)
    )
    grid = np.c_[xx1.ravel(), xx2.ravel()]
    Z = model.predict(grid).reshape(xx1.shape)

    plt.figure()
    plt.contourf(xx1, xx2, Z, alpha=0.25, levels=np.array([-0.1, 0.5, 1.1]))
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker='o', edgecolor='k', label='train')
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='^', edgecolor='k', label='test')
    plt.legend()
    plt.title("Decision Region — Logistic Regression (2 features)")
    plt.xlabel(feat_names[0] if feat_names else "x1")
    plt.ylabel(feat_names[1] if feat_names else "x2")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    _iris_binary_versicolor_virginica()
