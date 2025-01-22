import numpy as np
import matplotlib.pyplot as plt

# Problem 1: Creating random numbers
np.random.seed(0)  
mean_1 = [-3, 0]
cov_matrix = [[1.0, 0.8], [0.8, 1.0]]

data_1 = np.random.multivariate_normal(mean_1, cov_matrix, 500)

# Problem 2: Visualization using scatter plots
plt.figure(figsize=(8, 6))
plt.scatter(data_1[:, 0], data_1[:, 1], alpha=0.7, label="Data 1")
plt.title("Scatter Plot of Data 1")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.legend()
plt.grid()
plt.show()

# Problem 3: Visualization using histograms
plt.figure(figsize=(12, 6))

# Histogram for Dimension 1
plt.subplot(1, 2, 1)
plt.hist(data_1[:, 0], bins=30, alpha=0.7, color='blue', label="Dimension 1")
plt.title("Histogram of Dimension 1")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.xlim([-6, 0])
plt.grid()

# Histogram for Dimension 2
plt.subplot(1, 2, 2)
plt.hist(data_1[:, 1], bins=30, alpha=0.7, color='orange', label="Dimension 2")
plt.title("Histogram of Dimension 2")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.xlim([-4, 4])
plt.grid()

plt.tight_layout()
plt.show()

# Problem 4: Adding data
mean_2 = [0, -3] 
data_2 = np.random.multivariate_normal(mean_2, cov_matrix, 500)  

plt.figure(figsize=(8, 6))
plt.scatter(data_1[:, 0], data_1[:, 1], alpha=0.7, label="Data 1")
plt.scatter(data_2[:, 0], data_2[:, 1], alpha=0.7, label="Data 2", color="red")
plt.title("Scatter Plot of Data 1 and Data 2")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.legend()
plt.grid()
plt.show()

# Problem 5: Data Combining
combined_data = np.vstack((data_1, data_2))  
print(f"Combined data shape: {combined_data.shape}") 

# Problem 6: Labeling
labels_1 = np.zeros(data_1.shape[0])  
labels_2 = np.ones(data_2.shape[0])  
labels = np.concatenate((labels_1, labels_2)) 

labeled_data = np.column_stack((combined_data, labels))
print(f"Labeled data shape: {labeled_data.shape}")  

print("Sample labeled data:")
print(labeled_data[:5])  
