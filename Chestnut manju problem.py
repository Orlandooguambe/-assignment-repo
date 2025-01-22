import math
import matplotlib.pyplot as plt

# Constants
radius_bun = 0.05  # radius of the chestnut bun in meters
radius_solar_system = 4.5 * 10**12  # radius of the solar system in meters (approx. Neptune's orbit)

# Volume of a sphere: V = (4/3) * π * r^3
def volume_of_bun(radius):
    return (4/3) * math.pi * radius**3

# Volume of the solar system (approximated as a sphere)
def volume_of_solar_system(radius):
    return (4/3) * math.pi * radius**3

# Calculate the time it takes to exceed the solar system volume
def time_to_cover_solar_system(initial_volume_bun, solar_system_volume):
    t = 5 * math.log2(solar_system_volume / initial_volume_bun)  
    return t

# Calculate the initial volume of the chestnut bun
initial_volume_bun = volume_of_bun(radius_bun)

# Calculate the volume of the solar system
solar_system_volume = volume_of_solar_system(radius_solar_system)

# Calculate the time it takes to cover the solar system
time_to_cover = time_to_cover_solar_system(initial_volume_bun, solar_system_volume)

# Convert time to days and print the result
time_in_days = time_to_cover / 60 / 24
print(f"It will take approximately {time_in_days:.2f} days to cover the solar system with chestnut buns.")

time_points = range(0, int(time_to_cover), 5)  
volume_points = [initial_volume_bun * 2**(t / 5) for t in time_points]

plt.figure(figsize=(10, 6))
plt.plot(time_points, volume_points, label="Volume of Chestnut Buns")
plt.axhline(y=solar_system_volume, color='r', linestyle='--', label="Volume of Solar System")
plt.title("Volume of Chestnut Buns Over Time")
plt.xlabel("Time (minutes)")
plt.ylabel("Volume (cubic meters)")
plt.legend()
plt.grid(True)
plt.yscale('log')  
plt.show()
