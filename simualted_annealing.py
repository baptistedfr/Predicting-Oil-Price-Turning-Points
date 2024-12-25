import math
import random
import matplotlib.pyplot as plt
import seaborn as sns

class SimulatedAnnealing:
    def __init__(self, objective_function, initial_solution, temp_start, temp_end, cooling_rate, max_iter):
        """
        Initialize the Simulated Annealing algorithm.

        Parameters:
            objective_function (function): The function to minimize.
            initial_solution (any): Initial guess for the solution.
            temp_start (float): Starting temperature.
            temp_end (float): Ending temperature.
            cooling_rate (float): Rate at which temperature decreases.
            max_iter (int): Maximum iterations at each temperature level.
        """
        self.objective_function = objective_function
        self.current_solution = initial_solution
        self.current_value = objective_function(initial_solution)
        self.best_solution = initial_solution
        self.best_value = self.current_value
        self.temperature = temp_start
        self.temp_end = temp_end
        self.cooling_rate = cooling_rate
        self.max_iter = max_iter
        self.objective_values = []  # Store all objective values encountered

    def neighbor(self, solution):
        """Generate a neighbor solution by tweaking the current solution."""
        return solution + random.uniform(-1, 1)

    def acceptance_probability(self, delta):
        """Calculate the acceptance probability."""
        if delta < 0:
            return 1.0
        return math.exp(-delta / self.temperature)

    def run(self):
        """
        Execute the Simulated Annealing algorithm.

        Returns:
            tuple: Best solution and its objective value.
        """
        while self.temperature > self.temp_end:
            for _ in range(self.max_iter):
                candidate_solution = self.neighbor(self.current_solution)
                candidate_value = self.objective_function(candidate_solution)

                # Record the objective value
                self.objective_values.append(candidate_value)

                delta = candidate_value - self.current_value

                if self.acceptance_probability(delta) > random.random():
                    self.current_solution = candidate_solution
                    self.current_value = candidate_value

                    if self.current_value < self.best_value:
                        self.best_solution = self.current_solution
                        self.best_value = self.current_value

            self.temperature *= self.cooling_rate

        return self.best_solution, self.best_value

    def plot_distribution(self):
        """
        Plot the distribution of objective values encountered during the run.
        """
        sns.histplot(self.objective_values, kde=True, bins=50)
        plt.title("Distribution of Objective Function Values")
        plt.xlabel("Objective Value")
        plt.ylabel("Frequency")
        plt.show()

# Example usage
def objective_function(x):
    """A sample objective function: A simple quadratic function."""
    return x**2 + 4*math.sin(5*x)

initial_solution = random.uniform(-10, 10)
temp_start = 1000
temp_end = 1e-3
cooling_rate = 0.9
max_iter = 10

sa = SimulatedAnnealing(objective_function, initial_solution, temp_start, temp_end, cooling_rate, max_iter)
best_solution, best_value = sa.run()

print(f"Best Solution: {best_solution}")
print(f"Best Objective Value: {best_value}")

# Visualize the distribution of objective values
sa.plot_distribution()
