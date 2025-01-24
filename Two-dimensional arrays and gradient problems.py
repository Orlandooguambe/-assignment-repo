import numpy as np

x = np.arange(-50, 50.1, 0.1)
y = (1/2) * x + 1

print(f"x: {x[:5]}...{x[-5:]}")
print(f"y: {y[:5]}...{y[-5:]}")
 
 # x and y into a single 2D array
array_xy = np.column_stack((x, y))

# the shape and the first few elements
print(f"Shape of combined array: {array_xy.shape}")
print(f"First 5 rows of array:\n{array_xy[:5]}")

# the gradient using finite differences
dx = np.diff(x)  
dy = np.diff(y)  
gradient = dy / dx  # Gradient (dy/dx)

#  the results
print(f"First 5 gradients: {gradient[:5]}")
print(f"Gradient array length: {len(gradient)}") 
import matplotlib.pyplot as plt

# Plot the linear function
plt.figure(figsize=(10, 6))

# Plot y = (1/2)x + 1
plt.subplot(2, 1, 1)
plt.plot(x, y, label="y = 0.5x + 1", color='blue')
plt.title("Linear Function")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

# Plot gradient
plt.subplot(2, 1, 2)
plt.plot(x[:-1], gradient, label="Gradient (dy/dx)", color='red')
plt.title("Gradient of Linear Function")
plt.xlabel("x")
plt.ylabel("Gradient")
plt.legend()

plt.tight_layout()
plt.show()

def compute_gradient(function, x_range=(-50, 50.1, 0.1)):
     x = np.arange(*x_range)
     y = function(x)
     dx = np.diff(x)
     dy = np.diff(y)
     gradient = dy / dx
     array_xy = np.column_stack((x, y))
     return array_xy, gradient

# functions
def function1(x):
    return x**2

def function2(x):
    return 2 * x**2 + 2 * x

def function3(x):
    return np.sin(x / 12)

#  gradients for all functions
functions = [function1, function2, function3]
x_ranges = [(-50, 50.1, 0.1), (-50, 50.1, 0.1), (0, 50.1, 0.1)]

for i, (func, x_range) in enumerate(zip(functions, x_ranges), 1):
    array_xy, gradient = compute_gradient(func, x_range)
    x, y = array_xy[:, 0], array_xy[:, 1]
    
    # the function and its gradient
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(x, y, label=f"Function {i}", color='blue')
    plt.title(f"Function {i}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(x[:-1], gradient, label=f"Gradient of Function {i}", color='red')
    plt.title(f"Gradient of Function {i}")
    plt.xlabel("x")
    plt.ylabel("Gradient")
    plt.legend()

    plt.tight_layout()
    plt.show()

# y = x^2
array_xy, gradient = compute_gradient(function1)
x, y = array_xy[:, 0], array_xy[:, 1]

# the minimum value of y
min_y = y.min()
min_index = y.argmin()
min_x = x[min_index]

# Gradients before and after the minimum
gradient_before = gradient[min_index - 1] if min_index > 0 else None
gradient_after = gradient[min_index] if min_index < len(gradient) else None

print(f"Minimum y: {min_y}")
print(f"Corresponding x: {min_x}")
print(f"Gradient before minimum: {gradient_before}")
print(f"Gradient after minimum: {gradient_after}")
