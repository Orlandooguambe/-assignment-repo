import time
import matplotlib.pyplot as plt

# [Problem 1] Implementation using exponentiation arithmetic operators
THICKNESS = 0.00008  
n_folds = 43  
folded_thickness = THICKNESS * (2 ** n_folds)

# [Problem 2] Unit Conversion
print("Thickness: {:.2f} kilometers".format(folded_thickness / 1000))

# [Problem 3] Create using a for statement
THICKNESS = 0.00008  # Initial thickness in meters
n_folds = 43  # Number of folds

folded_thickness = THICKNESS
for _ in range(n_folds):
    folded_thickness *= 2

print("Thickness: {:.2f} kilometers".format(folded_thickness / 1000))

# [Problem 4] Comparison of calculation time (using time)
start = time.time()
folded_thickness_exp = THICKNESS * (2 ** n_folds)
elapsed_time_exp = time.time() - start
print("Exponentiation time (time): {:.8f} seconds".format(elapsed_time_exp))

start = time.time()
folded_thickness_loop = THICKNESS
for _ in range(n_folds):
    folded_thickness_loop *= 2
elapsed_time_loop = time.time() - start
print("For loop time (time): {:.8f} seconds".format(elapsed_time_loop))

# [Problem 4 extra] Comparison with %%timeit 
print("\n--- Timing with %%timeit (Jupyter magic) ---")
# In a notebook, run the following lines separately:
# %%timeit -n 1000
# THICKNESS * (2 ** n_folds)
#
# %%timeit -n 1000
# folded_thickness = THICKNESS
# for _ in range(n_folds):
#     folded_thickness *= 2

# [Problem 5] Saving to a list
thickness_list = [THICKNESS]
current_thickness = THICKNESS
for _ in range(n_folds):
    current_thickness *= 2
    thickness_list.append(current_thickness)
print("Number of values stored:", len(thickness_list))

# [Problem 6] Displaying a line graph
plt.title("Thickness of folded paper")
plt.xlabel("Number of folds")
plt.ylabel("Thickness (meters)")
plt.plot(thickness_list)
plt.show()

# [Problem 7] Customizing graphs
plt.figure(figsize=(10, 6))  
plt.title("Thickness of Folded Paper", fontsize=16)
plt.xlabel("Number of Folds", fontsize=14)
plt.ylabel("Thickness (meters)", fontsize=14)
plt.plot(thickness_list, color='green', linestyle='--', linewidth=2)
plt.tick_params(labelsize=12)  
plt.grid(True)  
plt.show()
