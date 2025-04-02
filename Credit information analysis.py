import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno

# Exercise 1: Understanding the competition
# Kaggle's Home Credit Default Risk competition aims to predict the likelihood of a client defaulting on a loan.
# Home Credit is a financial company that provides loans to individuals who may not have access to traditional banking.
# The competition helps improve risk assessment, leading to better loan decisions and financial inclusion.

# Exercise 2: Understanding the overview of data
# Load dataset
file_path = 'application_train.csv'  # Update with correct path
data = pd.read_csv(file_path)

# Display first few rows
data.head()

# General dataset information
data.info()

# Summary statistics
print(data.describe())

# Check for missing values
missing_values = data.isnull().sum()
missing_values = missing_values[missing_values > 0]
print("Missing values per column:")
print(missing_values)

# Visualizing missing values
msno.matrix(data)
plt.show()

# Plot distribution of target variable
sns.countplot(x='TARGET', data=data)
plt.title("Distribution of Loan Repayment Status")
plt.show()

# Exercise 3: Defining issues
# - What features have the most impact on loan default?
# - How do income levels relate to repayment ability?
# - Does employment length affect loan default rates?
# - How does age correlate with credit risk?
# - Are there missing values in key features, and how should they be handled?

# Exercise 4: Data exploration
# Generate insights through visualizations

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), cmap='coolwarm', annot=False)
plt.title("Feature Correlation Heatmap")
plt.show()

# Distribution of client's income
sns.histplot(data['AMT_INCOME_TOTAL'], bins=50, kde=True)
plt.title("Distribution of Client's Income")
plt.xlabel("Income")
plt.ylabel("Count")
plt.show()

# Relationship between age and loan default
sns.boxplot(x=data['TARGET'], y=data['DAYS_BIRTH'] / -365)
plt.title("Age Distribution by Loan Repayment Status")
plt.xlabel("Loan Default (1 = Yes, 0 = No)")
plt.ylabel("Age (Years)")
plt.show()

# Employment length vs default risk
sns.boxplot(x=data['TARGET'], y=data['DAYS_EMPLOYED'] / -365)
plt.title("Employment Length by Loan Repayment Status")
plt.xlabel("Loan Default (1 = Yes, 0 = No)")
plt.ylabel("Employment Length (Years)")
plt.show()

# Loan amounts distribution
sns.histplot(data['AMT_CREDIT'], bins=50, kde=True)
plt.title("Distribution of Loan Amounts")
plt.xlabel("Loan Amount")
plt.ylabel("Count")
plt.show()


