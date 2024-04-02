import numpy as np
import math
import random

# Constants
size = 2  # Lattice size
n = size * size  # Number of spin points on lattice
T = 5.0  # Starting temperature
minT = 0.5  # Minimum temperature
change = 0.1  # Temperature step size
mcs = 10000  # Number of Monte Carlo steps
transient = 1000  # Number of transient steps
norm = 1.0 / (mcs * n)  # Normalization for averaging

# Initialize lattice
def initialize():
    return np.random.choice([-1, 1], size=(size, size))

# Output lattice configuration
def output(lat):
    for row in lat:
        print(' '.join(['+' if s == 1 else '-' for s in row]))

# Calculate energy at a position
def energy_pos(lat, x, y):
    left = lat[x - 1][y] if x > 0 else lat[size - 1][y]
    right = lat[(x + 1) % size][y]
    up = lat[x][(y + 1) % size]
    down = lat[x][y - 1] if y > 0 else lat[x][size - 1]

    return -lat[x][y] * (left + right + up + down)

# Test for spin flip
def test_flip(lat, x, y, T):
    de = -2 * energy_pos(lat, x, y)
    if de < 0 or random.random() < math.exp(-de / T):
        return True
    return False

# Flip spin
def flip(lat, x, y):
    lat[x][y] *= -1

# Calculate total magnetization
def total_magnetization(lat):
    return np.sum(lat)

# Calculate total energy
def total_energy(lat):
    return sum(energy_pos(lat, x, y) for x in range(size) for y in range(size)) / 2

# Main program
def main():
    global T  # Define T as global to modify it inside the function
    lat = initialize()
    data = []

    current_temp = T  # Use a separate variable for the loop
    while current_temp >= minT:
        # Transient phase
        for _ in range(transient):
            x, y = np.random.randint(size), np.random.randint(size)
            flip(lat, x, y)

        # Observables calculation
        E, M = total_energy(lat), total_magnetization(lat)
        etot, etotsq, mtot, mtotsq = 0, 0, 0, 0

        # Monte Carlo loop
        for _ in range(mcs):
            for _ in range(n):
                x, y = np.random.randint(size), np.random.randint(size)
                if test_flip(lat, x, y, current_temp):
                    flip(lat, x, y)
                    E += 2 * energy_pos(lat, x, y)
                    M = total_magnetization(lat)

            etot += E
            etotsq += E**2
            mtot += M
            mtotsq += M**2

        # Averages
        E_avg = etot * norm
        Esq_avg = etotsq * norm
        M_avg = mtot * norm
        Msq_avg = mtotsq * norm

        # Append data
        data.append([current_temp, M_avg, Msq_avg, E_avg, Esq_avg])

        # Update temperature
        current_temp -= change

    # Write data to file
    with open('DATA.1.dat', 'w') as f:
        for row in data:
            f.write('\t'.join(map(str, row)) + '\n')

if __name__ == "__main__":
    main()
