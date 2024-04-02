import numpy as np

# Global Constants
kB = 1  # Boltzmann constant in reduced units
J = 1   # Exchange energy

def ising_simulate(steps, size, temperature, J=1, H=0):
    spins = np.random.choice([-1, 1], size=(size, size))
    for step in range(steps):
        spins = monte_carlo_step(size, temperature, spins, J, H)
    energy, magnetization = calculate_observables(J, H, spins, size)
    specific_heat = calculate_specific_heat(spins, temperature, size)
    susceptibility = calculate_susceptibility(spins, temperature, size)
    return energy, magnetization, specific_heat, susceptibility

def monte_carlo_step(size, temperature, spins, J, H):
    """Perform a single Monte Carlo move."""
    for step in range(size ** 2):
        i, j = np.random.randint(0, size, 2)  # Random spin
        delta_E = delta_energy(i, j, spins, size, J, H)

        # Metropolis algorithm
        if delta_E < 0 or np.random.rand() < np.exp(-delta_E / (kB * temperature)):
            spins[i, j] *= -1  # Flip spin
    return spins

def delta_energy(i, j, spins, size, J, H):
    """Calculate the change in energy if spin (i, j) is flipped."""
    # Periodic boundary conditions
    up = spins[(i - 1) % size, j]
    down = spins[(i + 1) % size, j]
    left = spins[i, (j - 1) % size]
    right = spins[i, (j + 1) % size]

    # Change in energy
    delta_E = 2 * spins[i, j] * (J * (up + down + left + right) + H)
    return delta_E

def calculate_observables(J, H, spins, size):
    """Calculate and return observables like energy, magnetization."""
    energy = -J * (np.sum(spins * np.roll(spins, 1, axis=0)) + np.sum(spins * np.roll(spins, 1, axis=1)))
    energy -= H * np.sum(spins)
    magnetization = np.sum(spins)
    return energy, magnetization

def calculate_specific_heat(spins, temperature, size):
    """Calculate the specific heat capacity."""
    energy_sq = np.sum(spins * np.roll(spins, 1, axis=0)) + np.sum(spins * np.roll(spins, 1, axis=1))
    energy_sq *= -J
    energy_sq = energy_sq ** 2
    avg_energy_sq = np.mean(energy_sq)
    energy = -J * (np.sum(spins * np.roll(spins, 1, axis=0)) + np.sum(spins * np.roll(spins, 1, axis=1)))
    avg_energy = np.mean(energy)
    specific_heat = (avg_energy_sq - avg_energy ** 2) / (kB * temperature ** 2 * size ** 2)
    return specific_heat

def calculate_susceptibility(spins, temperature, size):
    """Calculate the magnetic susceptibility."""
    magnetization = np.sum(spins)
    magnetization_sq = magnetization ** 2
    avg_magnetization_sq = np.mean(magnetization_sq)
    avg_magnetization = np.mean(magnetization)
    susceptibility = (avg_magnetization_sq - avg_magnetization ** 2) / (kB * temperature * size ** 2)
    return susceptibility

# input
size = 10  # 10x10 lattice
temperature = 2.5  # Temperature (in units where kB=1)
steps = 1000  # Number of Monte Carlo steps

energy, magnetization, specific_heat, susceptibility = ising_simulate(steps, size, temperature)
print(f"Energy: {energy}")
print(f"Magnetization: {magnetization}")
print(f"Specific Heat: {specific_heat}")
print(f"Susceptibility: {susceptibility}")
