import numpy as np

class IsingModel:
    def __init__(self, size, temperature, J=1, H=0):
        self.size = size  # Size of the lattice, N x N
        self.temperature = temperature  # Temperature
        self.J = J  # Interaction strength
        self.H = H  # External magnetic field
        self.spins = np.random.choice([-1, 1], size=(size, size))  # Spin lattice

    def delta_energy(self, i, j):
        """Calculate the change in energy if spin (i, j) is flipped."""
        # Periodic boundary conditions
        top = self.spins[(i - 1) % self.size, j]
        bottom = self.spins[(i + 1) % self.size, j]
        left = self.spins[i, (j - 1) % self.size]
        right = self.spins[i, (j + 1) % self.size]

        # Change in energy
        delta_E = 2 * self.J * self.spins[i, j] * (top + bottom + left + right)
        return delta_E

    def monte_carlo_step(self):
        """Perform a single Monte Carlo move."""
        for _ in range(self.size ** 2):
            i, j = np.random.randint(0, self.size, 2)  # Random spin
            delta_E = self.delta_energy(i, j)

            # Metropolis algorithm
            if delta_E < 0 or np.random.rand() < np.exp(-delta_E / self.temperature):
                self.spins[i, j] *= -1  # Flip spin

    def simulate(self, steps):
        """Run the Monte Carlo simulation for a given number of steps."""
        for _ in range(steps):
            self.monte_carlo_step()

    def calculate_observables(self):
        """Calculate and return observables like energy, magnetization."""
        energy = -self.J * np.sum(self.spins * np.roll(self.spins, 1, axis=0))
        energy -= self.J * np.sum(self.spins * np.roll(self.spins, 1, axis=1))
        energy -= self.H * np.sum(self.spins)

        magnetization = np.sum(self.spins)

        return energy, magnetization

    def calculate_specific_heat(self):
        """Calculate the specific heat capacity."""
        energy_sqrd = np.sum(self.calculate_observables()[0] ** 2)
        avg_energy_sqrd = energy_sqrd / self.size ** 2
        avg_energy = np.sum(self.calculate_observables()[0]) / self.size ** 2
        specific_heat = (avg_energy_sqrd - avg_energy ** 2) / (self.temperature ** 2)
        return specific_heat

    def calculate_susceptibility(self):
        """Calculate the magnetic susceptibility."""
        magnetization = self.calculate_observables()[1]
        magnetization_sqrd = magnetization ** 2
        avg_magnetization_sqrd = magnetization_sqrd / self.size ** 2
        avg_magnetization = magnetization / self.size ** 2
        susceptibility = (avg_magnetization_sqrd - avg_magnetization ** 2) / self.temperature
        return susceptibility


# Example usage:
size = 10  # 10x10 lattice
temperature = 2.5  # Temperature (in units where kB=1)
steps = 1000  # Number of Monte Carlo steps

ising_model = IsingModel(size, temperature)
ising_model.simulate(steps)
energy, magnetization = ising_model.calculate_observables()
specific_heat = ising_model.calculate_specific_heat()
susceptibility = ising_model.calculate_susceptibility()

print(f"Energy: {energy}")
print(f"Magnetization: {magnetization}")
print(f"Specific Heat: {specific_heat}")
print(f"Susceptibility: {susceptibility}")
