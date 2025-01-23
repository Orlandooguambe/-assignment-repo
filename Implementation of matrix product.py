import numpy as np

# Define matrices
a_ndarray = np.array([[-1, 2, 3], [4, -5, 6], [7, 8, -9]])
b_ndarray = np.array([[0, 2, 1], [0, 2, -8], [2, 9, -1]])

# Matrix multiplication
c_ndarray = np.matmul(a_ndarray, b_ndarray)
print(c_ndarray)



# Compute a specific element of the matrix product
i, j = 0, 0  # Specify row and column
result = sum(a_ndarray[i, k] * b_ndarray[k, j] for k in range(3))
print(result)  # Should print 6




def matrix_product(a, b):
    # Ensure the matrices are compatible for multiplication
    rows_a, cols_a = a.shape
    rows_b, cols_b = b.shape
    if cols_a != rows_b:
        raise ValueError("Matrix dimensions do not align for multiplication")

    # Initialize result matrix with zeros
    result = np.zeros((rows_a, cols_b))
    for i in range(rows_a):
        for j in range(cols_b):
            result[i, j] = sum(a[i, k] * b[k, j] for k in range(cols_a))
    return result

# Call the function
c_manual = matrix_product(a_ndarray, b_ndarray)
print(c_manual)



def safe_matrix_product(a, b):
    # Ensure the matrices are compatible for multiplication
    rows_a, cols_a = a.shape
    rows_b, cols_b = b.shape
    if cols_a != rows_b:
        return f"Cannot multiply matrices: A is {rows_a}x{cols_a}, B is {rows_b}x{cols_b}."
    return matrix_product(a, b)

# Example with incompatible matrices
d_ndarray = np.array([[-1, 2, 3], [4, -5, 6]])
e_ndarray = np.array([[-9, 8, 7], [6, -5, 4]])
print(safe_matrix_product(d_ndarray, e_ndarray))



# Transpose B and calculate
b_transposed = b_ndarray.T
c_transposed = np.matmul(a_ndarray, b_transposed)
print(c_transposed)
