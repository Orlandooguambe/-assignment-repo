import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Problem 1: Data acquisition
dataset = load_iris()
X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
y = pd.DataFrame(dataset.target, columns=["Species"])

# Problem 2: Data combination
df = pd.concat([X, y], axis=1)

# Problem 3: Data verification
print(df.head())  # Display the first rows
df.info()  # Data structure
print(df.isnull().sum())  # Check for missing values
print(df.describe())  # Basic statistics
print(df['Species'].value_counts())  # Count of each class

# Problem 4: Research about the dataset
# The Iris dataset was introduced by Ronald Fisher in 1936 in his study on linear discriminant analysis. 
# The data was collected from a pasture on the Gaspé Peninsula, Canada.
# 
# - The dataset consists of 150 samples from three species of iris flowers: Setosa, Versicolor, and Virginica (50 samples each).
# - Each sample has four features:
#   - sepal_length (sepal length)
#   - sepal_width (sepal width)
#   - petal_length (petal length)
#   - petal_width (petal width)
# 
# - The dataset is widely used in machine learning and statistics, especially for classification tasks.
# - It is a classic example of how to distinguish different groups based on numerical characteristics.
# 
# - Importance in machine learning:
#   - The Iris dataset is clean and well-balanced, making it ideal for testing classification algorithms such as k-NN, logistic regression, and decision trees.
#   - It is often used to teach concepts like data visualization, classification, and correlation analysis.

# Problem 5: Data extraction
print(df['sepal width (cm)'])  # Extraction by column name
print(df.loc[:, 'sepal width (cm)'])  # Another extraction method
print(df.iloc[50:100])  # Extract data between indices 50-99
print(df.loc[50:99, 'petal length (cm)'])  # Specific extraction
print(df[df['petal width (cm)'] == 0.2])  # Filtering by value

# Problem 6: Creating charts
# Pie chart
df['Species'].value_counts().plot.pie(autopct='%1.1f%%', figsize=(6, 6))
plt.title("Species Distribution")
plt.show()

# Boxplot and Violin plot for each feature
for feature in dataset.feature_names:
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=df['Species'], y=df[feature])
    plt.title(f'Boxplot of {feature}')
    plt.show()
    
    sns.violinplot(x=df['Species'], y=df[feature])
    plt.title(f'Violin Plot of {feature}')
    plt.show()

# Problem 7: Relationship between features
sns.pairplot(df, hue='Species')  # Scatterplot matrix
plt.show()

# Correlation matrix and Heatmap
correlation_matrix = df.corr()
print(correlation_matrix)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# Problem 8: Explanation of graphs
# - The species are well separated in the graphs
# - Petal length and petal width are good discriminators between classes
# - Setosa has distinctly different values from the other two species
