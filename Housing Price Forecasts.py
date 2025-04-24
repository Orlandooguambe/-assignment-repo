import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# 1. Load data
file_path = "train.csv"  # update with your path if needed
data = pd.read_csv(file_path)

# 2. Select features and target
features = data[["GrLivArea", "YearBuilt"]]
target = data["SalePrice"]

# 3. Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# 4. Initialize models
models = {
    "Linear Regression": LinearRegression(),
    "SVM": SVR(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor()
}

# 5. Train, predict, evaluate, visualize
results = {}
plt.figure(figsize=(16, 10))
for i, (name, model) in enumerate(models.items(), 1):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    results[name] = mse

    # Plot
    plt.subplot(2, 2, i)
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{name}\nMSE: {mse:.2f}")

plt.tight_layout()
plt.show()

# 6. Summary table
results_df = pd.DataFrame.from_dict(results, orient='index', columns=['MSE'])
print("\nModel Performance (MSE):")
print(results_df.sort_values(by="MSE"))
