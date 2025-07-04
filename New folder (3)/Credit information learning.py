import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation

# 1. Load data
train_df = pd.read_csv("application_train.csv")
test_df = pd.read_csv("application_test.csv")

# Separate target
y = train_df["TARGET"]
X = train_df.drop(columns=["TARGET", "SK_ID_CURR"])
test_id = test_df["SK_ID_CURR"]
X_test = test_df.drop(columns=["SK_ID_CURR"])

# Standard preprocessing function
def preprocess(df):
    df = df.copy()
    df = df.fillna(-999)
    for col in df.select_dtypes(include="object").columns:
        df[col], _ = pd.factorize(df[col])
    return df

# Feature engineering base
def create_features(df, version):
    df = preprocess(df)

    if version == 1:
        # Original version (no extra features)
        return df

    elif version == 2:
        # Add income-to-credit ratio
        df["INCOME_CREDIT_RATIO"] = df["AMT_INCOME_TOTAL"] / (df["AMT_CREDIT"] + 1)

    elif version == 3:
        # Add normalized employment duration
        df["DAYS_EMPLOYED_PERC"] = df["DAYS_EMPLOYED"] / (df["DAYS_BIRTH"] + 1)

    elif version == 4:
        # Sum of documents provided
        doc_cols = [col for col in df.columns if "FLAG_DOCUMENT" in col]
        df["TOTAL_DOCS"] = df[doc_cols].sum(axis=1)

    elif version == 5:
        # Add credit-to-annuity ratio
        df["CREDIT_ANNUITY_RATIO"] = df["AMT_CREDIT"] / (df["AMT_ANNUITY"] + 1)

    return df

# Loop through 5 feature versions
results = []
for version in range(1, 6):
    print(f"\n📦 Training feature version {version}...")

    X_feat = create_features(X, version)
    X_test_feat = create_features(X_test, version)

    # Train/Validation split
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_feat, y, test_size=0.2, random_state=42)

    # Model
    model = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.05, random_state=42)
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[early_stopping(100), log_evaluation(100)]
    )

    # Evaluation
    y_pred = model.predict_proba(X_valid)[:, 1]
    auc = roc_auc_score(y_valid, y_pred)
    print(f"✅ AUC (version {version}): {auc:.4f}")
    results.append((version, auc, model, X_test_feat))

# Show ranking
results = sorted(results, key=lambda x: x[1], reverse=True)
print("\n🏆 Ranking of feature versions:")
for v, auc, _, _ in results:
    print(f"Version {v} - AUC: {auc:.4f}")

# 5. Use best version for prediction and submission
best_version, best_auc, best_model, best_test = results[0]

print(f"\n🚀 Using version {best_version} for submission (AUC: {best_auc:.4f})")

# Prediction
test_preds = best_model.predict_proba(best_test)[:, 1]
submission = pd.DataFrame({
    "SK_ID_CURR": test_id,
    "TARGET": test_preds
})
submission.to_csv("submission_best.csv", index=False)
print("\n📁 Submission saved as 'submisssion_best.csv'")
