import numpy as np
import matplotlib.pyplot as plt
import datetime
import time

kB = 1  # Boltzmann constant in reduced units

def simulate(steps, size, temperature, J=1., H=0):
    spins = np.random.choice([-1, 1], size=(size, size)) #generates the spins on the latice
    # initalized as floats to avoid overflow errors
    sum_energy = 0.
    sum_energy_squared = 0.
    sum_magnetization = 0.
    sum_magnetization_squared = 0.

    for step in range(steps): # full steps loop
        spins = montecarloMetStep(size, temperature, spins, J, H)
        energy = hamiltonian(J, H, spins)
        magnetization = np.sum(spins)

        #data collection
        sum_energy += energy
        sum_energy_squared += energy ** 2
        sum_magnetization += magnetization
        sum_magnetization_squared += magnetization ** 2

    # data averaging
    avg_energy = sum_energy / steps
    avg_magnetization = sum_magnetization / steps

    heatcapacity = calculateHeatCapacity(steps, temperature, avg_energy, sum_energy_squared)
    MagneticSus = calculateMagneticSus(steps, temperature, avg_magnetization, sum_magnetization_squared)
    return avg_energy, avg_magnetization, heatcapacity, MagneticSus

def calculateHeatCapacity(steps, temperature, avg_energy, sum_energy_squared):
    avg_energy_squared = sum_energy_squared / steps
    heatcapacity = (avg_energy_squared - avg_energy ** 2) / (kB * temperature ** 2)
    return heatcapacity

def calculateMagneticSus(steps, temperature, avg_magnetization, sum_magnetization_squared):
    avg_magnetization_squared = sum_magnetization_squared / steps
    MagneticSus = (avg_magnetization_squared - avg_magnetization ** 2) / (kB * temperature)
    return MagneticSus

# def simulate(steps, size, temperature, J=1, H=0):
#     spins = np.random.choice([-1, 1], size=(size, size))
#     for step in range(steps):
#         spins = montecarloMetStep(size, temperature, spins, J, H)
#     magnetization = np.sum(spins)
#     energy = hamiltonian(J, H, spins)
#     #energy -= (H * np.sum(spins))
#     heatcapacity = calculateHeatCapacity(J, H, spins, temperature)
#     MagneticSus = calculateMagneticSus(spins, temperature)
#     return energy, magnetization, heatcapacity, MagneticSus

def montecarloMetStep(size, temperature, spins, J, H):
    for step in range(size ** 2):
        i, j = np.random.randint(0, size, 2)
        dEng = dEnergy(spins, i, j, size, J, H)
        if dEng < 0 or np.random.rand() < np.exp(-dEng / (kB * temperature)):
                spins[i, j] *= -1
    return spins

def dEnergy(spins, i, j, size, J, H):
    up = spins[(i - 1) % size, j]
    down = spins[(i + 1) % size, j]
    left = spins[i, (j - 1) % size]
    right = spins[i, (j + 1) % size]

    dEng = 2 * spins[i, j] * (J * (up + down + left + right) + H)
    return dEng

def hamiltonian(J, H, spins):
    energy = (-J * (np.sum(spins * np.roll(spins, 1, axis=0)) + np.sum(spins * np.roll(spins, 1, axis=1)))) -(H * np.sum(spins))
    return energy

def plotModel(size, steps, J=1, H=0):
    start_time = time.time() # getting the start time
    energies = []
    magnetizations = []
    heat_capacities = []
    magnetic_susceptibilities = []
    temperatures = []

    tempRange = np.linspace(1.0, 4.0, 60)  # temp range from 1-4, creates 60 data points
    inverseTotalTemps = 1 / len(tempRange)  # inverse of the total temp values

    for index, temp in enumerate(tempRange):
        percent_complete = (index + 1) * inverseTotalTemps * 100  # calculates the progress of the program
        print(f'Percent complete: {percent_complete:.2f}%')  # used to display the progress of the program when running for a long period of time

        energy, magnetization, heat_capacity, magnetic_sus = simulate(steps, size, temp, J, H)  # calling the simulation
        # adds the data points to the corresponding lists
        energies.append(energy)
        magnetizations.append(abs(magnetization))
        heat_capacities.append(heat_capacity)
        magnetic_susceptibilities.append(magnetic_sus)
        temperatures.append(temp)

    plt.figure(figsize=(12, 10))

    # Plot energy
    plt.subplot(2, 2, 1)
    plt.plot(temperatures, energies, 'o-')
    plt.title('Energy')
    plt.xlabel('Temperature')
    plt.ylabel('Energy')

    # Plot magnetization
    plt.subplot(2, 2, 2)
    plt.plot(temperatures, magnetizations, 'o-')
    plt.title('Magnetization')
    plt.xlabel('Temperature')
    plt.ylabel('Magnetization')

    # Plot heat capacity
    plt.subplot(2, 2, 3)
    plt.plot(temperatures, heat_capacities, 'o-')
    plt.title('Heat Capacity')
    plt.xlabel('Temperature')
    plt.ylabel('Heat Capacity')

    # Plot magnetic susceptibility
    plt.subplot(2, 2, 4)
    plt.plot(temperatures, magnetic_susceptibilities, 'o-')
    plt.title('Magnetic Susceptibility')
    plt.xlabel('Temperature')
    plt.ylabel('Magnetic Susceptibility')

    end_time = time.time() # getting the end time
    print(end_time - start_time) # displaying the program run time
    # formating, saving and displaying the plotted data to the screen
    plt.tight_layout()
    plotname = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    #plt.savefig(fr'C:\Users\Tommy\PycharmProjects\IsingMonteCarlo\output\{plotname}.png')
    plt.show()

if __name__ == '__main__':
    size = 10  # 10x10 lattice
    steps = 2000  # Monte Carlo steps

    plotModel(size, steps)