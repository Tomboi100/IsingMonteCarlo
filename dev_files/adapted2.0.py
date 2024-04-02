import numpy as np
import random
import matplotlib.pyplot as plt

# Constants
size = 5 # Lattice size, larger for realistic phase transition simulation
J = 1.0  # Interaction strength
kb = 1.0  # Boltzmann constant in reduced units
T_start = 5.0  # Starting temperature
T_end = 0.5  # Minimum temperature
dT = 0.1  # Temperature step size
mcs = 10000  # Number of Monte Carlo steps
transient = 1000  # Number of transient steps
norm = 1.0 / (mcs * size * size)  # Normalization for averaging
H = 0  # External magnetic field


# Initialize lattice
def initialize():
    return np.random.choice([-1, 1], size=(size, size))


# Calculate energy of a given configuration
def total_energy(lat, H):
    energy = -J * (sum(sum(lat[i] * lat[(i + 1) % size, j] for i in range(size)) for j in range(size)) +
                   sum(sum(lat[i, j] * lat[i, (j + 1) % size] for i in range(size)) for j in range(size)))
    # Magnetic field term
    mag_energy = -H * np.sum(lat)
    return energy + mag_energy


# Calculate total magnetization
def total_magnetization(lat):
    return np.sum(lat)


# Calculate delta energy for flipping a spin at (x, y)
def delta_energy(lat, x, y, H):
    # Periodic boundary conditions
    left = lat[x - 1][y] if x > 0 else lat[-1][y]
    right = lat[(x + 1) % size][y]
    up = lat[x][(y + 1) % size]
    down = lat[x][y - 1] if y > 0 else lat[x][-1]

    # Change in energy if we flip spin s[x][y]
    delta_e = 2 * J * lat[x][y] * (left + right + up + down)
    # Magnetic field term
    delta_m = 2 * H * lat[x][y]
    return delta_e + delta_m


# Monte Carlo step
def monte_carlo_step(lat, T, H):
    for _ in range(size * size):
        x, y = random.randint(0, size - 1), random.randint(0, size - 1)
        de = delta_energy(lat, x, y, H)
        if de < 0 or random.random() < np.exp(-de / (kb * T)):
            lat[x][y] *= -1


# Main simulation
def simulate():
    lat = initialize()
    T = T_start
    temperatures = []
    energies = []
    magnetizations = []
    heat_capacities = []
    susceptibilities = []

    while T >= T_end:
        # Transient phase
        for _ in range(transient):
            monte_carlo_step(lat, T, H)

        # Equilibration phase
        e_data = []
        m_data = []
        for _ in range(mcs):
            monte_carlo_step(lat, T, H)
            e = total_energy(lat, H)
            m = total_magnetization(lat)
            e_data.append(e)
            m_data.append(m)

        # Calculate observables
        E_avg = np.mean(e_data)
        M_avg = np.mean(m_data)
        E_sq_avg = np.mean(np.square(e_data))
        M_sq_avg = np.mean(np.square(m_data))

        # Calculate heat capacity and susceptibility
        C = (E_sq_avg - E_avg ** 2) * norm / (kb * T ** 2)
        X = (M_sq_avg - M_avg ** 2) * norm / (kb * T)

        # Store results
        temperatures.append(T)
        energies.append(E_avg)
        magnetizations.append(M_avg)
        heat_capacities.append(C)
        susceptibilities.append(X)

        # Decrease temperature
        T -= dT

    # Plot results using Matplotlib
    plt.figure(figsize=(12, 8))

    # Plot the average energy per spin
    plt.subplot(2, 2, 1)
    plt.plot(temperatures, energies, 'o-')
    plt.title('Average Energy per Spin')
    plt.xlabel('Temperature (T)')
    plt.ylabel('Average Energy')

    # Plot the average magnetization per spin
    plt.subplot(2, 2, 2)
    plt.plot(temperatures, magnetizations, 'o-')
    plt.title('Average Magnetization per Spin')
    plt.xlabel('Temperature (T)')
    plt.ylabel('Average Magnetization')

    # Plot the heat capacity per spin
    plt.subplot(2, 2, 3)
    plt.plot(temperatures, heat_capacities, 'o-')
    plt.title('Heat Capacity per Spin')
    plt.xlabel('Temperature (T)')
    plt.ylabel('Heat Capacity')

    # Plot the magnetic susceptibility per spin
    plt.subplot(2, 2, 4)
    plt.plot(temperatures, susceptibilities, 'o-')
    plt.title('Magnetic Susceptibility per Spin')
    plt.xlabel('Temperature (T)')
    plt.ylabel('Susceptibility')

    plt.tight_layout()  # Adjust subplots to fit into the figure area.
    plt.show()  # Display the plots

if __name__ == '__main__':
    simulate()