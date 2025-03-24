import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from scipy.stats import kurtosis, skew

# Exercise 1: Obtaining the dataset
file_path = 'train.csv'  # Update with the correct path
data = pd.read_csv(file_path)

# Exercise 2: Investigating the dataset itself
# General overview of the data
data.info()
print(data.describe())

# Exercise 3: Checking the data
# Check for missing values
missing_values = data.isnull().sum()
missing_values = missing_values[missing_values > 0]
print("Missing values per column:")
print(missing_values)

# Visualizing missing values
msno.matrix(data)
plt.show()

# Exercise 4: Dealing with missing values
# Remove columns with more than 5 missing values
data_cleaned = data.dropna(axis=1, thresh=len(data) - 5)

# Remove rows with remaining missing values
data_cleaned = data_cleaned.dropna()
print("Cleaned dataset:")
data_cleaned.info()

# Exercise 6: Objective variable
# Define the target variable
target = 'SalePrice'

# Visualizing the distribution of the target variable
sns.displot(data[target], kde=True)
plt.title("Distribution of SalePrice")
plt.show()

# Calculate kurtosis and skewness
kurt = kurtosis(data[target])
skew_val = skew(data[target])
print(f"Kurtosis: {kurt}, Skewness: {skew_val}")

# Apply logarithmic transformation
data[target] = np.log1p(data[target])

# Visualizing the distribution after transformation
sns.displot(data[target], kde=True)
plt.title("Distribution of SalePrice after log transformation")
plt.show()

# Recalculate kurtosis and skewness
kurt_log = kurtosis(data[target])
skew_log = skew(data[target])
print(f"Kurtosis after log: {kurt_log}, Skewness after log: {skew_log}")

# Exercise 7: Confirming the correlation coefficient
# Select only numerical columns
numerical_data = data.select_dtypes(include=['number'])

# Create a correlation matrix
corr_matrix = numerical_data.corr()

# Create a heatmap of the correlation matrix
top_10_features = corr_matrix[target].sort_values(ascending=False).head(11).index
plt.figure(figsize=(10, 8))
sns.heatmap(numerical_data[top_10_features].corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title("Heatmap of the 10 most correlated variables with SalePrice")
plt.show()

# Identify highly correlated features
high_corr_features = corr_matrix[target].loc[abs(corr_matrix[target]) > 0.8]
print("Highly correlated features with SalePrice:")
print(high_corr_features)
