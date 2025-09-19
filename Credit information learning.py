import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation

# 1) Load data
train_df = pd.read_csv("application_train.csv")
test_df  = pd.read_csv("application_test.csv")

y = train_df["TARGET"].astype(int)
train_df = train_df.drop(columns=["TARGET"])

train_id = train_df["SK_ID_CURR"]
test_id  = test_df["SK_ID_CURR"]

# -------- Helpers --------
def _factorize_joint(train_like: pd.DataFrame, test_like: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One consistent factorization for object columns by concatenating train+test."""
    tr = train_like.copy()
    te = test_like.copy()
    cat_cols = tr.select_dtypes(include="object").columns.tolist()
    if not cat_cols:
        return tr, te
    joint = pd.concat([tr[cat_cols], te[cat_cols]], axis=0)
    for c in cat_cols:
        # Use pandas Categorical codes to keep stable mapping
        cats = joint[c].astype("category").cat.categories
        tr[c] = pd.Categorical(tr[c], categories=cats).codes
        te[c] = pd.Categorical(te[c], categories=cats).codes
    return tr, te

def preprocess(train_like: pd.DataFrame, test_like: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Impute simple, encode categóricas de forma consistente."""
    tr = train_like.copy()
    te = test_like.copy()

    # Tratar sentinela famoso da competição
    if "DAYS_EMPLOYED" in tr.columns:
        tr.loc[tr["DAYS_EMPLOYED"] == 365243, "DAYS_EMPLOYED"] = np.nan
    if "DAYS_EMPLOYED" in te.columns:
        te.loc[te["DAYS_EMPLOYED"] == 365243, "DAYS_EMPLOYED"] = np.nan

    # Factorização consistente
    tr, te = _factorize_joint(tr, te)

    # Imputação simples para numéricos e categóricos (após factorizar)
    tr = tr.fillna(-999)
    te = te.fillna(-999)

    return tr, te

def create_features(train_like: pd.DataFrame, test_like: pd.DataFrame, version: int):
    """Gera as 5 variações de features de forma consistente entre train e test."""
    tr = train_like.copy()
    te = test_like.copy()

    # Base preprocess (sem novas colunas)
    tr, te = preprocess(tr, te)

    if version == 1:
        # baseline sem features extras
        return tr, te

    elif version == 2:
        # income-to-credit
        for df in (tr, te):
            if {"AMT_INCOME_TOTAL","AMT_CREDIT"}.issubset(df.columns):
                df["INCOME_CREDIT_RATIO"] = df["AMT_INCOME_TOTAL"] / (df["AMT_CREDIT"] + 1e-9)
        return tr, te

    elif version == 3:
        # emprego/idade – usar anos negativos como estão na base original (ambos são negativos)
        for df in (tr, te):
            if {"DAYS_EMPLOYED","DAYS_BIRTH"}.issubset(df.columns):
                df["DAYS_EMPLOYED_PERC"] = df["DAYS_EMPLOYED"] / (df["DAYS_BIRTH"] + 1e-9)
        return tr, te

    elif version == 4:
        # soma de documentos
        doc_cols = [c for c in tr.columns if "FLAG_DOCUMENT" in c]
        for df in (tr, te):
            if doc_cols:
                df["TOTAL_DOCS"] = df[doc_cols].sum(axis=1)
        return tr, te

    elif version == 5:
        # credit-to-annuity
        for df in (tr, te):
            if {"AMT_CREDIT","AMT_ANNUITY"}.issubset(df.columns):
                df["CREDIT_ANNUITY_RATIO"] = df["AMT_CREDIT"] / (df["AMT_ANNUITY"] + 1e-9)
        return tr, te

    # fallback
    return tr, te

# ---------- Loop de versões ----------
X_base = train_df.drop(columns=["SK_ID_CURR"])
X_test_base = test_df.drop(columns=["SK_ID_CURR"])

results = []
for version in range(1, 6):
    print(f"\n📦 Training feature version {version}...")

    X_feat, X_test_feat = create_features(X_base, X_test_base, version)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_feat, y, test_size=0.2, stratify=y, random_state=42
    )

    model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        random_state=42,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[early_stopping(100), log_evaluation(100)]
    )

    y_pred = model.predict_proba(X_valid)[:, 1]
    auc = roc_auc_score(y_valid, y_pred)
    print(f"✅ AUC (version {version}): {auc:.4f}")

    results.append((version, auc, model, X_test_feat))

# Ranking
results = sorted(results, key=lambda x: x[1], reverse=True)
print("\n🏆 Ranking of feature versions:")
for v, auc, _, _ in results:
    print(f"Version {v} - AUC: {auc:.4f}")

# Melhor versão para submissão
best_version, best_auc, best_model, best_test = results[0]
print(f"\n🚀 Using version {best_version} for submission (AUC: {best_auc:.4f})")

test_preds = best_model.predict_proba(best_test)[:, 1]
submission = pd.DataFrame({"SK_ID_CURR": test_id, "TARGET": test_preds})
submission.to_csv("submission_best.csv", index=False)
print("📁 Submission saved as 'submission_best.csv'")
