import numpy as np
import timeit

def hamiltonian_without_roll(J, H, spins):
    size = spins.shape[0]
    energy = 0
    for i in range(size):
        for j in range(size):
            up = spins[i, (j-1) % size]
            down = spins[i, (j+1) % size]
            left = spins[(i-1) % size, j]
            right = spins[(i+1) % size, j]
            energy -= J * spins[i, j] * (up + down + left + right)
    energy -= H * np.sum(spins)
    return energy

def hamiltonian_roll(J, H, spins):
    energy = (-J * (np.sum(spins * np.roll(spins, 1, axis=0)) + np.sum(spins * np.roll(spins, 1, axis=1)))) - (
                H * np.sum(spins))
    return energy

def hamiltonian_custom(J, H, spins):
    size = spins.shape[0]
    i, j = np.random.randint(0, size, 2)
    energy = 0
    up = spins[i, (j - 1) % size]
    down = spins[i, (j + 1) % size]
    left = spins[(i - 1) % size, j]
    right = spins[(i + 1) % size, j]
    # Calculate interaction energy (sum over all pairs of adjacent spins)
    interaction_energy = -J * (np.sum(spins * up) + np.sum(spins * down) + np.sum(spins * left) + np.sum(spins * right))

    # Magnetic field contribution
    magnetic_field_energy = -H * np.sum(spins)

    # Total energy
    energy = interaction_energy + magnetic_field_energy
    return energy

def hamiltonian_slice(J, H, spins):
    # Get the size of the lattice
    size = spins.shape[0]

    # Create shifted arrays for each direction using slicing for periodic boundary conditions
    up = np.vstack((spins[1:], spins[0]))
    down = np.vstack((spins[-1], spins[:-1]))
    left = np.hstack((spins[:, 1:], spins[:, 0:1]))
    right = np.hstack((spins[:, -1:], spins[:, :-1]))

    # Calculate interaction energy (sum over all pairs of adjacent spins)
    interaction_energy = -J * (np.sum(spins * up) + np.sum(spins * down) + np.sum(spins * left) + np.sum(spins * right))

    # Magnetic field contribution
    magnetic_field_energy = -H * np.sum(spins)

    # Total energy
    energy = interaction_energy + magnetic_field_energy
    return energy

size = 10
# Example lattice
spins = np.random.choice([-1, 1], size=(size, size))

# Timing the two functions
time_roll = timeit.timeit(lambda: hamiltonian_roll(1, 0, spins), number=100)
time_slice = timeit.timeit(lambda: hamiltonian_slice(1, 0, spins), number=100)
#time_without = timeit.timeit(lambda: hamiltonian_without_roll(1, 0, spins), number=100)
time_without = timeit.timeit(lambda: hamiltonian_custom(1, 0, spins), number=100)

print(f"Time with np.roll: {time_roll} seconds")
print(f"Time with slicing: {time_slice} seconds")
print(f"Time with slicing: {time_without} seconds")