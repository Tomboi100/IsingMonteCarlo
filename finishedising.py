import numpy as np
import matplotlib.pyplot as plt
import time
import datetime

kB = 1  # Boltzmann constant in reduced units

# main simulation function
def simulate(steps, size, temperature, J=1., H=0):
    spins = np.random.choice([-1, 1], size=(size, size)) #generates the random spins on the latice
    # initalized summation variables as floats to avoid overflow errors
    sum_energy = 0.
    sum_energy_squared = 0.
    sum_magnetization = 0.
    sum_magnetization_squared = 0.

    # setting the initial calculations for energy and magnetization
    energy = hamiltonian(J, H, spins)
    magnetization = np.sum(spins)

    # Equilibration
    for step in range(steps//10): # bias removal step at 10% of the number of steps
        spins, energy, magnetization = montecarloMetStep(size, temperature, spins, J, H, energy, magnetization)

    # Main Monte Carlo simulation
    for step in range(steps): # full steps loop
        spins, energy, magnetization = montecarloMetStep(size, temperature, spins, J, H, energy, magnetization)

        # data collection
        sum_energy += energy
        sum_energy_squared += energy ** 2
        sum_magnetization += magnetization
        sum_magnetization_squared += magnetization ** 2

    # data averaging from collected data, multiplies by the inverse to save on computation
    inverseSteps = 1/steps
    avg_energy = sum_energy * inverseSteps
    avg_magnetization = sum_magnetization * inverseSteps

    # calling functions to calculate heat capacity and magnetic susceptibility
    heatcapacity = calculateHeatCapacity(inverseSteps, temperature, avg_energy, sum_energy_squared)
    MagneticSus = calculateMagneticSus(inverseSteps, temperature, avg_magnetization, sum_magnetization_squared)
    return avg_energy, avg_magnetization, heatcapacity, MagneticSus

# function to calculate the heat capacity of the system
def calculateHeatCapacity(inversesteps, temperature, avg_energy, sum_energy_squared):
    avg_energy_squared = sum_energy_squared * inversesteps
    heatcapacity = (avg_energy_squared - avg_energy ** 2) / (kB * temperature ** 2) # C = dU/dT = (1/Kb T**2)*((E**2)-(E)**2)
    return heatcapacity

# function to calculate the magnetic sus
def calculateMagneticSus(inversesteps, temperature, avg_magnetization, sum_magnetization_squared):
    avg_magnetization_squared = sum_magnetization_squared * inversesteps
    MagneticSus = (avg_magnetization_squared - avg_magnetization ** 2) / (kB * temperature) # X = dM/dH = (1/Kb T)*((E**2)-(E)**2)
    return MagneticSus

# Monte Carlo with Metropolis step function
def montecarloMetStep(size, temperature, spins, J, H, Eng, Mag):
    i, j = np.random.randint(0, size, 2) # randomly select the spin
    dEng = dEnergy(spins, i, j, size, J, H) # call change in energy calculation function

    #Metropolis algorithm (previous version had  **if dEng = 0**)
    if dEng <= 0 or np.random.rand() < np.exp(-dEng / (kB * temperature)): # statement to check if the move should be accepted
            spins[i, j] *= -1 # flip the spin
            # update the energy and magnetization
            Eng += dEng
            Mag += 2*spins[i,j]
    return spins, Eng, Mag

# function to calculate the change in energy
def dEnergy(spins, i, j, size, J, H):
    # getting values from the nearest cells in the lattice with periodic boundary conditions
    up = spins[(i - 1) % size, j]
    down = spins[(i + 1) % size, j]
    left = spins[i, (j - 1) % size]
    right = spins[i, (j + 1) % size]

    dEng = 2 * spins[i, j] * (J * (up + down + left + right) + H) # change in energy from spin flip
    return dEng

def hamiltonian(J, H, spins):
    #peridoic boundary conditions using np.roll, np.roll is less memory efficent but is faster than other methods of calculating ΔEnergy
    energy = (-J * (np.sum(spins * np.roll(spins, 1, axis=0)) + np.sum(spins * np.roll(spins, 1, axis=1)))) -(H * np.sum(spins))
    return energy

# visual data plotting function
def plotModel(size, steps, J=1, H=0):
    start_time = time.time() # getting the start time
    # initalized empty lists to append the values collected
    energies = []
    magnetizations = []
    heat_capacities = []
    magnetic_susceptibilities = []
    temperatures = []

    tempRange = np.linspace(1.0, 4.0, 60)  # temp range from 1-4, creates 60 data points

    for temp in tempRange:
        energy, magnetization, heat_capacity, magnetic_sus = simulate(steps, size, temp, J, H) # calling the simulation
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
    print(fr'This program took: {end_time - start_time} seconds to run') # displaying the program run time

    # formating, saving and displaying the plotted data to the screen
    plt.tight_layout()
    plotname = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    plt.savefig(fr'C:\Users\Tommy\PycharmProjects\IsingMonteCarlo\output\{plotname}.png')
    plt.show()

if __name__ == '__main__':
    # input data for size and number of steps
    size = 10  # 10x10 lattice
    steps = 200000  # Monte Carlo steps 31577seconds 8.7hours for 20000000 steps 5217seconds for 2000000, 7904
    # curie temperature should be around 2.5

    # run simulation and plotting
    plotModel(size, steps)