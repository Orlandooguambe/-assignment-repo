#[Question 1] Number of grains of rice on the 100th day and graph
import matplotlib.pyplot as plt

def compute_sorori_shinzaemon(n_days=100):
   
    list_n_grains = [1]  # Grains received on day 1
    list_total_grains = [1]  # Cumulative total on day 1

    for day in range(1, n_days):
        grains_today = list_n_grains[-1] * 2  # Double the grains from the previous day
        list_n_grains.append(grains_today)
        list_total_grains.append(list_total_grains[-1] + grains_today)

    return list_n_grains, list_total_grains


list_n_grains, list_total_grains = compute_sorori_shinzaemon(n_days=100)

print(f"Total grains of rice on the 100th day: {list_total_grains[-1]:,}")
plt.figure(figsize=(12, 6))
plt.plot(range(1, 101), list_n_grains, label="Grains received per day")
plt.plot(range(1, 101), list_total_grains, label="Cumulative grains received", linestyle="--")
plt.title("Rice grains received and accumulated over days", fontsize=16)
plt.xlabel("Days", fontsize=14)
plt.ylabel("Number of rice grains", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()

# [Question 2] Function with a variable number of days
n_days = 30  
list_n_grains, list_total_grains = compute_sorori_shinzaemon(n_days=n_days)
print(f"Total grains of rice in {n_days} days: {list_total_grains[-1]:,}")
plt.figure(figsize=(12, 6))
plt.plot(range(1, n_days + 1), list_n_grains, label="Grains received per day")
plt.plot(range(1, n_days + 1), list_total_grains, label="Cumulative grains received", linestyle="--")
plt.title(f"Rice grains received and accumulated in {n_days} days", fontsize=16)
plt.xlabel("Days", fontsize=14)
plt.ylabel("Number of rice grains", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()

#Question 3] How many people can survive on the rice they receive for how many days?

def calculate_survival_days(total_grains, num_people):
    grains_per_day_per_person = 19200  
    grains_per_day = grains_per_day_per_person * num_people
    days = total_grains // grains_per_day
    return days
num_people = 10  
total_grains = list_total_grains[-1]  
days = calculate_survival_days(total_grains, num_people)
print(f"With {num_people} people, the rice lasts for {days} days.")
