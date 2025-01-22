import math

# Problem 1: Minimum folds to exceed the height of Mt. Fuji
def folds_to_exceed_height(target_height, t0=0.00008):
    n = math.ceil(math.log2(target_height / t0))
    return n

# Problem 2: Function to find the number of folds required to exceed any height
def folds_for_arbitrary_height(target_height, t0=0.00008):
    return folds_to_exceed_height(target_height, t0)

# Problem 3: Length of paper required for a given number of folds
def paper_length_for_folds(n, t0=0.00008):
    # Calculate the length required to fold t0 thick paper n times
    length = (math.pi * t0 / 6) * (2**n + 4) * (2**n - 1)
    return length

# Testing Problem 1 with Mt. Fuji (3776m)
mt_fuji_height = 3776
n_mt_fuji = folds_to_exceed_height(mt_fuji_height)
print(f"Minimum folds to exceed Mt. Fuji height: {n_mt_fuji}")

# Testing Problem 2 with Proxima Centauri (4.0175 x 10^16 m)
proxima_centauri_distance = 4.0175 * 10**16
n_proxima = folds_for_arbitrary_height(proxima_centauri_distance)
print(f"Minimum folds to reach Proxima Centauri: {n_proxima}")

# Testing Problem 3: Length of paper for the Moon, Mt. Fuji, and Proxima Centauri
moon_distance = 384400000  # Distance to the Moon in meters
mt_fuji_length = paper_length_for_folds(n_mt_fuji)
proxima_length = paper_length_for_folds(n_proxima)

print(f"Length of paper to reach the Moon: {paper_length_for_folds(folds_for_arbitrary_height(moon_distance))} meters")
print(f"Length of paper to exceed Mt. Fuji: {mt_fuji_length} meters")
print(f"Length of paper to reach Proxima Centauri: {proxima_length} meters")
