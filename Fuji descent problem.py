import numpy as np
import matplotlib.pyplot as plt

csv_path = "mtfuji_data.csv"
np.set_printoptions(suppress=True)
fuji = np.loadtxt(csv_path, delimiter=",", skiprows=1)

point_numbers = fuji[:, 0]
elevation = fuji[:, 3]  

def visualize_elevation(point_numbers, elevation):
    plt.figure(figsize=(10, 6))
    plt.plot(point_numbers, elevation, label='Elevation')
    plt.title('Cross-Section of Mt. Fuji')
    plt.xlabel('Point Number')
    plt.ylabel('Elevation (m)')
    plt.legend()
    plt.grid()
    plt.show()

visualize_elevation(point_numbers, elevation)

def calculate_gradient(current_point, elevation):
    if current_point == 0:  
        return 0
    gradient = (elevation[current_point] - elevation[current_point - 1]) / 1  
    return gradient


def calculate_next_point(current_point, gradient, alpha=0.2):
    next_point = current_point - alpha * gradient
    next_point = int(round(next_point))
    
    next_point = max(0, min(next_point, len(elevation) - 1))
    return next_point


def descend_mountain(initial_point, elevation, alpha=0.2):
    current_point = initial_point
    path = [current_point]
    
    while True:
        gradient = calculate_gradient(current_point, elevation)
        next_point = calculate_next_point(current_point, gradient, alpha)
        if next_point == current_point:  
            break
        path.append(next_point)
        current_point = next_point
    
    return path


def visualize_descent(initial_point, elevation, path):
    plt.figure(figsize=(10, 6))
    plt.plot(point_numbers, elevation, label='Elevation')
    plt.scatter(path, elevation[path], color='red', label='Descent Path')
    plt.title(f'Descent Process from Point {initial_point}')
    plt.xlabel('Point Number')
    plt.ylabel('Elevation (m)')
    plt.legend()
    plt.grid()
    plt.show()

initial_point = 136
path = descend_mountain(initial_point, elevation)
visualize_descent(initial_point, elevation, path)


all_paths = {i: descend_mountain(i, elevation) for i in range(len(elevation))}

def visualize_multiple_descents(initial_points, elevation):
    plt.figure(figsize=(10, 6))
    plt.plot(point_numbers, elevation, label='Elevation')
    
    for point in initial_points:
        path = descend_mountain(point, elevation)
        plt.scatter(path, elevation[path], label=f'Start: {point}')
    
    plt.title('Descent Process for Multiple Starting Points')
    plt.xlabel('Point Number')
    plt.ylabel('Elevation (m)')
    plt.legend()
    plt.grid()
    plt.show()

visualize_multiple_descents([50, 100, 136, 200], elevation)



def visualize_alpha_effect(initial_point, elevation, alphas):
    plt.figure(figsize=(10, 6))
    plt.plot(point_numbers, elevation, label='Elevation')
    
    for alpha in alphas:
        path = descend_mountain(initial_point, elevation, alpha=alpha)
        plt.scatter(path, elevation[path], label=f'α={alpha}')
    
    plt.title(f'Effect of α on Descent from Point {initial_point}')
    plt.xlabel('Point Number')
    plt.ylabel('Elevation (m)')
    plt.legend()
    plt.grid()
    plt.show()


visualize_alpha_effect(136, elevation, [0.1, 0.2, 0.5, 1.0])
