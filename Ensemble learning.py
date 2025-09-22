 
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor


# --------------------------
# Data loading and splitting
# --------------------------

def load_houseprices_xy(csv_path: str = "train.csv",
                        features=("GrLivArea", "YearBuilt"),
                        target="SalePrice",
                        drop_na=True) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    cols = list(features) + [target]
    df = df[cols]
    if drop_na:
        df = df.dropna()
    X = df[list(features)].values.astype(float)
    y = df[target].values.astype(float)
    return X, y


# --------------------------
# Blending (simple)
# --------------------------

def fit_base_models(X_tr, y_tr):
    """
    Build a diverse set of base regressors.
    - Different algorithms
    - Different preprocessing
    - Different hyperparameters
    """
    models = []

    # Linear Regression (no scaling)
    models.append(("linreg", LinearRegression()))

    # Ridge with polynomial features (degree 2)
    models.append((
        "poly2_ridge",
        Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=10.0, random_state=42))
        ])
    ))

    # SVR with RBF kernel (scale inputs)
    models.append((
        "svr_rbf",
        Pipeline([
            ("scaler", StandardScaler()),
            ("svr", SVR(kernel="rbf", C=10.0, epsilon=0.1, gamma="scale"))
        ])
    ))

    # Decision Tree (depth-limited)
    models.append(("tree", DecisionTreeRegressor(max_depth=5, random_state=42)))

    # kNN Regressor (scaled)
    models.append((
        "knn",
        Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsRegressor(n_neighbors=7, weights="distance"))
        ])
    ))

    # Fit all
    fitted = []
    for name, model in models:
        m = clone(model)
        m.fit(X_tr, y_tr)
        fitted.append((name, m))
    return fitted


def blend_predictions(models, X) -> np.ndarray:
    """
    Equal-weight average of predictions from models.
    """
    preds = [m.predict(X) for _, m in models]
    return np.mean(np.column_stack(preds), axis=1)


def weighted_blend_predictions(models, X, weights: List[float]) -> np.ndarray:
    """
    Weighted average (weights sum to 1 ideally).
    """
    preds = [m.predict(X) for _, m in models]
    P = np.column_stack(preds)
    w = np.array(weights, dtype=float).reshape(-1, 1)
    return (P @ w).ravel()


# --------------------------
# Bagging (scratch)
# --------------------------

@dataclass
class BaggingRegressorScratch:
    base_estimator: object
    n_estimators: int = 25
    max_samples: float = 1.0   # fraction of training set
    bootstrap: bool = True
    random_state: int = 42

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        n = X.shape[0]
        m = int(np.ceil(self.max_samples * n))
        self.estimators_ = []
        for i in range(self.n_estimators):
            if self.bootstrap:
                idx = rng.integers(0, n, size=m)
            else:
                idx = rng.choice(n, size=m, replace=False)
            est = clone(self.base_estimator)
            est.fit(X[idx], y[idx])
            self.estimators_.append(est)
        return self

    def predict(self, X):
        preds = np.column_stack([est.predict(X) for est in self.estimators_])
        return np.mean(preds, axis=1)


# --------------------------
# Stacking (scratch)
# --------------------------

@dataclass
class StackingRegressorScratch:
    base_models: List[Tuple[str, object]]
    meta_model: object
    n_folds: int = 5
    shuffle: bool = True
    random_state: int = 42

    def fit(self, X, y):
        X = np.asarray(X); y = np.asarray(y)
        kf = KFold(n_splits=self.n_folds, shuffle=self.shuffle, random_state=self.random_state)

        # Level-0 out-of-fold predictions
        oof_pred_list = []
        self.base_fitted_ = []
        for name, model in self.base_models:
            oof = np.zeros_like(y, dtype=float)
            for tr_idx, va_idx in kf.split(X):
                m = clone(model)
                m.fit(X[tr_idx], y[tr_idx])
                oof[va_idx] = m.predict(X[va_idx])
            oof_pred_list.append(oof)
            # fit model on full data for test-time use later
            m_full = clone(model).fit(X, y)
            self.base_fitted_.append((name, m_full))

        Z = np.column_stack(oof_pred_list)   # (n_samples, n_base)
        self.meta_model_ = clone(self.meta_model).fit(Z, y)
        return self

    def predict(self, X):
        # level-0 predictions on new data
        base_preds = [m.predict(X) for _, m in self.base_fitted_]
        Z_test = np.column_stack(base_preds)
        return self.meta_model_.predict(Z_test)


# --------------------------
# Runner
# --------------------------

def evaluate_mse(y_true, y_pred, label):
    mse = mean_squared_error(y_true, y_pred)
    print(f"{label:<28s} MSE: {mse:,.2f}")
    return mse


def main():
    # 1) Load data
    X, y = load_houseprices_xy("train.csv",
                               features=("GrLivArea", "YearBuilt"),
                               target="SalePrice",
                               drop_na=True)

    # Optional: try log target for stability (comment out to use raw)
    use_log_target = False
    if use_log_target:
        y_trans = np.log1p(y)
    else:
        y_trans = y

    # 2) Train/validation split (80/20)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y_trans, test_size=0.2, random_state=42
    )

    # 3) Single models (baselines)
    baselines = [
        ("LinearRegression", LinearRegression()),
        ("SVR_rbf", Pipeline([("scaler", StandardScaler()),
                              ("svr", SVR(kernel="rbf", C=10.0, epsilon=0.1, gamma="scale"))])),
        ("DecisionTree", DecisionTreeRegressor(max_depth=5, random_state=42)),
    ]
    fitted_baselines = []
    print("\n=== Single Models ===")
    for name, model in baselines:
        m = clone(model).fit(X_tr, y_tr)
        y_hat = m.predict(X_va)
        evaluate_mse(y_va, y_hat, name)
        fitted_baselines.append((name, m))

    # 4) Blending: diverse base models
    print("\n=== Blending ===")
    blend_models = fit_base_models(X_tr, y_tr)  # a diverse set
    # equal-weight blend
    y_blend_eq = blend_predictions(blend_models, X_va)
    evaluate_mse(y_va, y_blend_eq, "Blend (equal weights)")

    # a simple weighted blend example (tune weights by hand or grid)
    # here a light bias toward models that often do well on tabular small sets
    weights = np.array([0.20, 0.25, 0.25, 0.15, 0.15])  # sum to 1
    y_blend_w = weighted_blend_predictions(blend_models, X_va, weights)
    evaluate_mse(y_va, y_blend_w, "Blend (weighted)")

    # learning the blend weights with a meta linear reg on train predictions
    # (still "blending", not full stacking, because we fit weights on train preds)
    P_tr = np.column_stack([m.predict(X_tr) for _, m in blend_models])
    w_learner = LinearRegression().fit(P_tr, y_tr)
    y_blend_learned = np.column_stack([m.predict(X_va) for _, m in blend_models]) @ w_learner.coef_ + w_learner.intercept_
    evaluate_mse(y_va, y_blend_learned, "Blend (learned weights)")

    # 5) Bagging (scratch): bag a weak model (trees are typical for bagging)
    print("\n=== Bagging (scratch) ===")
    bag = BaggingRegressorScratch(
        base_estimator=DecisionTreeRegressor(max_depth=4, random_state=0),
        n_estimators=50,
        max_samples=0.8,
        bootstrap=True,
        random_state=123
    ).fit(X_tr, y_tr)
    y_bag = bag.predict(X_va)
    evaluate_mse(y_va, y_bag, "Bagging(DecisionTree)")

    # 6) Stacking (scratch): use a few base models + linear meta-model
    print("\n=== Stacking (scratch) ===")
    base_for_stack = [
        ("linreg", LinearRegression()),
        ("svr", Pipeline([("scaler", StandardScaler()),
                          ("svr", SVR(kernel="rbf", C=10.0, epsilon=0.1))])),
        ("tree", DecisionTreeRegressor(max_depth=5, random_state=42)),
        ("knn", Pipeline([("scaler", StandardScaler()),
                          ("knn", KNeighborsRegressor(n_neighbors=7, weights="distance"))])),
    ]
    stacker = StackingRegressorScratch(
        base_models=base_for_stack,
        meta_model=LinearRegression(),
        n_folds=5,
        shuffle=True,
        random_state=42
    ).fit(X_tr, y_tr)
    y_stack = stacker.predict(X_va)
    evaluate_mse(y_va, y_stack, "Stacking (meta=LinearReg)")

    # If we used log target, map predictions back to price scale for human reading
    if use_log_target:
        print("\nNote: results above are in log1p space. Example back-transform of one model:")
        example = np.expm1(y_blend_eq[:5])
        print("First 5 predictions back-transformed:", np.round(example, 2))


if __name__ == "__main__":
    main()
