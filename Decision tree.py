#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


# =========================
# Metrics for classification
# =========================

def gini_impurity(y):
    """
    [Problem 1] Gini impurity of a node.
    y : (n_samples,) integer labels (0..K-1)
    """
    if y.size == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return 1.0 - np.sum(p ** 2)


def information_gain(y_parent, y_left, y_right):
    """
    [Problem 2] Information Gain using Gini impurity.
    IG = I(parent) - (nL/nP)*I(left) - (nR/nP)*I(right)
    """
    nP = y_parent.size
    nL = y_left.size
    nR = y_right.size
    if nP == 0:
        return 0.0
    I_parent = gini_impurity(y_parent)
    I_left = gini_impurity(y_left)
    I_right = gini_impurity(y_right)
    return I_parent - (nL / nP) * I_left - (nR / nP) * I_right


# =========================================
# Depth 1 Decision Tree (Decision Stump)
# =========================================

class ScratchDecisionTreeClassifierDepth1:
    """
    [Problem 3 & 4] Depth-1 decision tree (single split) from scratch.

    Parameters
    ----------
    verbose : bool
        If True, prints best split and IG.

    Attributes (after fit)
    ----------
    feature_index_ : int
        Index of feature used for the split.
    threshold_ : float
        Threshold used for the split: go left if x[f] <= threshold else right.
    left_class_ : int
        Predicted class for the left leaf.
    right_class_ : int
        Predicted class for the right leaf.
    best_ig_ : float
        Information gain of the chosen split.
    """

    def __init__(self, verbose=False):
        self.verbose = bool(verbose)
        self.feature_index_ = None
        self.threshold_ = None
        self.left_class_ = None
        self.right_class_ = None
        self.best_ig_ = 0.0

    def _majority_class(self, y):
        vals, counts = np.unique(y, return_counts=True)
        return int(vals[np.argmax(counts)])

    def fit(self, X, y):
        """
        Learn the best single split across all features and thresholds.

        X : (n_samples, n_features)
        y : (n_samples,)
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel().astype(int)
        n, d = X.shape

        # If node is pure or too small, make it a leaf-only stump
        if gini_impurity(y) == 0.0 or n == 0:
            self.feature_index_ = 0
            self.threshold_ = np.inf  # no sample goes left
            self.left_class_ = self._majority_class(y)
            self.right_class_ = self._majority_class(y)
            self.best_ig_ = 0.0
            if self.verbose:
                print("[Depth1] Node already pure. Making trivial stump.")
            return self

        best_ig = -np.inf
        best_feat = None
        best_thr = None
        best_left_class = None
        best_right_class = None

        # Try every feature and every unique value as threshold
        for f in range(d):
            values = np.unique(X[:, f])
            for thr in values:
                left_mask = X[:, f] <= thr
                right_mask = ~left_mask
                y_left = y[left_mask]
                y_right = y[right_mask]
                if y_left.size == 0 or y_right.size == 0:
                    continue
                ig = information_gain(y, y_left, y_right)
                if ig > best_ig:
                    best_ig = ig
                    best_feat = f
                    best_thr = thr
                    best_left_class = self._majority_class(y_left)
                    best_right_class = self._majority_class(y_right)

        # If no split improved IG (pathological), fallback to single leaf
        if best_feat is None:
            self.feature_index_ = 0
            self.threshold_ = np.inf
            self.left_class_ = self._majority_class(y)
            self.right_class_ = self._majority_class(y)
            self.best_ig_ = 0.0
            if self.verbose:
                print("[Depth1] No valid split found. Trivial stump.")
            return self

        self.feature_index_ = best_feat
        self.threshold_ = float(best_thr)
        self.left_class_ = int(best_left_class)
        self.right_class_ = int(best_right_class)
        self.best_ig_ = float(best_ig)

        if self.verbose:
            print(f"[Depth1] Best split: feature={self.feature_index_}, "
                  f"threshold={self.threshold_:.6f}, IG={self.best_ig_:.6f}")
        return self

    def predict(self, X):
        """
        Route each sample left/right and return the leaf class.
        """
        X = np.asarray(X, dtype=float)
        f = self.feature_index_
        thr = self.threshold_
        left_mask = X[:, f] <= thr
        yhat = np.empty(X.shape[0], dtype=int)
        yhat[left_mask] = self.left_class_
        yhat[~left_mask] = self.right_class_
        return yhat


# =========================================
# Advanced: Depth-2 Decision Tree (optional)
# =========================================

class ScratchDecisionTreeClassifierDepth2:
    """
    Depth-2 decision tree: two levels of splitting.

    Implementation: learn a top-level stump, then fit one stump per child.
    """

    def __init__(self, verbose=False):
        self.verbose = bool(verbose)
        self.top_ = ScratchDecisionTreeClassifierDepth1(verbose=verbose)
        self.left_child_ = None  # stump or None
        self.right_child_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel().astype(int)

        # Fit top split
        self.top_.fit(X, y)

        # Partition data
        mask_left = X[:, self.top_.feature_index_] <= self.top_.threshold_
        X_L, y_L = X[mask_left], y[mask_left]
        X_R, y_R = X[~mask_left], y[~mask_left]

        # If child node is not pure, fit a stump on that child
        if y_L.size > 0 and gini_impurity(y_L) > 0.0:
            self.left_child_ = ScratchDecisionTreeClassifierDepth1(verbose=self.verbose)
            self.left_child_.fit(X_L, y_L)
        if y_R.size > 0 and gini_impurity(y_R) > 0.0:
            self.right_child_ = ScratchDecisionTreeClassifierDepth1(verbose=self.verbose)
            self.right_child_.fit(X_R, y_R)

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        # First route by top split
        mask_left = X[:, self.top_.feature_index_] <= self.top_.threshold_
        yhat = np.empty(X.shape[0], dtype=int)

        # Left branch
        if self.left_child_ is None:
            yhat[mask_left] = self.top_.left_class_
        else:
            yhat[mask_left] = self.left_child_.predict(X[mask_left])

        # Right branch
        if self.right_child_ is None:
            yhat[~mask_left] = self.top_.right_class_
        else:
            yhat[~mask_left] = self.right_child_.predict(X[~mask_left])

        return yhat


# =========================================
# Advanced: Unlimited-depth Decision Tree
# =========================================

class _Node:
    """Internal node for the recursive tree."""
    __slots__ = ("feature", "threshold", "left", "right", "leaf_class")

    def __init__(self):
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.leaf_class = None


class ScratchDecisionTreeClassifierDepthInf:
    """
    Unbounded-depth Gini tree with stopping criteria:
    - max_depth (optional)
    - min_samples_split (optional)
    - stop when gini == 0 or no split improves IG

    Parameters
    ----------
    max_depth : int or None
    min_samples_split : int
    verbose : bool
    """

    def __init__(self, max_depth=None, min_samples_split=2, verbose=False):
        self.max_depth = None if max_depth is None else int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.verbose = bool(verbose)
        self.root_ = None

    def _majority(self, y):
        vals, counts = np.unique(y, return_counts=True)
        return int(vals[np.argmax(counts)])

    def _best_split(self, X, y):
        n, d = X.shape
        best = (-np.inf, None, None)  # (IG, feature, threshold)
        for f in range(d):
            values = np.unique(X[:, f])
            for thr in values:
                left_mask = X[:, f] <= thr
                right_mask = ~left_mask
                yL, yR = y[left_mask], y[right_mask]
                if yL.size == 0 or yR.size == 0:
                    continue
                ig = information_gain(y, yL, yR)
                if ig > best[0]:
                    best = (ig, f, thr)
        return best  # (ig, f, thr)

    def _build(self, X, y, depth):
        node = _Node()

        # stopping conditions
        if gini_impurity(y) == 0.0 or X.shape[0] < self.min_samples_split or \
           (self.max_depth is not None and depth >= self.max_depth):
            node.leaf_class = self._majority(y)
            return node

        ig, f, thr = self._best_split(X, y)
        if ig <= 0.0 or f is None:
            node.leaf_class = self._majority(y)
            return node

        node.feature = f
        node.threshold = float(thr)

        left_mask = X[:, f] <= thr
        right_mask = ~left_mask

        node.left = self._build(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build(X[right_mask], y[right_mask], depth + 1)
        return node

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel().astype(int)
        self.root_ = self._build(X, y, depth=0)
        return self

    def _predict_one(self, x, node):
        if node.leaf_class is not None:
            return node.leaf_class
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(row, self.root_) for row in X], dtype=int)


# =========================
# Plotting decision regions
# =========================

def plot_decision_region_2d(model, X, y, title="Decision region"):
    """
    Visualize decision region for a 2D dataset.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).ravel().astype(int)
    x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx1, xx2 = np.meshgrid(
        np.linspace(x1_min, x1_max, 301),
        np.linspace(x2_min, x2_max, 301)
    )
    grid = np.c_[xx1.ravel(), xx2.ravel()]
    Z = model.predict(grid).reshape(xx1.shape)

    plt.figure()
    plt.contourf(xx1, xx2, Z, alpha=0.25, levels=[-0.1, 0.5, 1.1])
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c="tab:blue", edgecolor="k", label="class 0")
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c="tab:orange", edgecolor="k", label="class 1")
    plt.legend()
    plt.title(title)
    plt.xlabel("x1"); plt.ylabel("x2")
    plt.grid(True)
    plt.show()


# =========================
# Demo / Verification code
# =========================

def _make_simple_dataset_2(n_per_class=70, seed=0):
    """
    Synthetic 2D binary dataset similar to a simple teachable set.
    """
    rng = np.random.default_rng(seed)
    mean0, mean1 = np.array([-1.0, 0.0]), np.array([1.2, 0.4])
    cov0 = np.array([[0.5, 0.1], [0.1, 0.5]])
    cov1 = np.array([[0.6, -0.15], [-0.15, 0.6]])
    X0 = rng.multivariate_normal(mean0, cov0, size=n_per_class)
    X1 = rng.multivariate_normal(mean1, cov1, size=n_per_class)
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n_per_class, dtype=int), np.ones(n_per_class, dtype=int)])
    return X, y


def _evaluate_and_compare(model, X_tr, y_tr, X_te, y_te, title):
    print(f"=== {title} ===")
    yhat = model.predict(X_te)
    acc = (yhat == y_te).mean()
    print(f"Accuracy: {acc:.4f}")

    # Optional comparison to sklearn
    try:
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.metrics import accuracy_score, precision_score, recall_score
        sk = DecisionTreeClassifier(max_depth=None)  # full CART
        sk.fit(X_tr, y_tr)
        y_sk = sk.predict(X_te)
        print(f"sklearn Accuracy : {accuracy_score(y_te, y_sk):.4f}")
        print(f"sklearn Precision: {precision_score(y_te, y_sk, zero_division=0):.4f}")
        print(f"sklearn Recall   : {recall_score(y_te, y_sk, zero_division=0):.4f}")
    except Exception as e:
        print("sklearn comparison skipped:", e)


def demo_all():
    # Make data & split
    from sklearn.model_selection import train_test_split
    X, y = _make_simple_dataset_2(n_per_class=80, seed=42)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.35, random_state=1, stratify=y)

    # ----- Depth 1 -----
    stump = ScratchDecisionTreeClassifierDepth1(verbose=True)
    stump.fit(X_tr, y_tr)
    _evaluate_and_compare(stump, X_tr, y_tr, X_te, y_te, "Depth-1 Decision Stump")
    plot_decision_region_2d(stump, X_te, y_te, title="Decision region — Depth-1")

    # ----- Depth 2 -----
    tree2 = ScratchDecisionTreeClassifierDepth2(verbose=True)
    tree2.fit(X_tr, y_tr)
    _evaluate_and_compare(tree2, X_tr, y_tr, X_te, y_te, "Depth-2 Decision Tree")
    plot_decision_region_2d(tree2, X_te, y_te, title="Decision region — Depth-2")

    # ----- Unlimited depth (with simple stopping) -----
    tree_inf = ScratchDecisionTreeClassifierDepthInf(max_depth=None, min_samples_split=2, verbose=False)
    tree_inf.fit(X_tr, y_tr)
    _evaluate_and_compare(tree_inf, X_tr, y_tr, X_te, y_te, "Depth-Inf Decision Tree")
    plot_decision_region_2d(tree_inf, X_te, y_te, title="Decision region — Depth-Inf")


if __name__ == "__main__":
    # Run the demo
    demo_all()
