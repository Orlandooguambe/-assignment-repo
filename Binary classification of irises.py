import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Task 1: Load the dataset and select features and categories
from sklearn.datasets import load_iris
dataset = load_iris()
data = pd.DataFrame(dataset.data, columns=dataset.feature_names)
data['species'] = dataset.target

# Select only virginica (label 2) and versicolor (label 1) with two features
binary_data = data[data['species'].isin([1, 2])]
X = binary_data[['sepal length (cm)', 'petal length (cm)']].values
y = binary_data['species'].values

# Task 2: Data analysis - Visualizations
sns.pairplot(binary_data, hue='species')
plt.show()

sns.boxplot(x=binary_data['species'], y=binary_data['sepal length (cm)'])
plt.show()

sns.violinplot(x=binary_data['species'], y=binary_data['petal length (cm)'])
plt.show()

# Task 3: Split data into training and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Task 4: Preprocessing and standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Task 5: Training and estimation with k-NN classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Task 6: Evaluation
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, pos_label=2)
rec = recall_score(y_test, y_pred, pos_label=2)
f1 = f1_score(y_test, y_pred, pos_label=2)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {acc:.2f}")
print(f"Precision: {prec:.2f}")
print(f"Recall: {rec:.2f}")
print(f"F1 Score: {f1:.2f}")
print("Confusion Matrix:")
print(conf_matrix)

# Task 7: Visualization of decision boundary
def decision_region(X, y, model, title='Decision Region'):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k')
    plt.title(title)
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Petal Length (cm)')
    plt.show()

decision_region(X_test, y_test, knn, "Decision Boundary of 3-NN")
