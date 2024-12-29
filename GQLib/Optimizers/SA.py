from .abstract_optimizers import Optimizer
from GQLib.LPPL import LPPL
from typing import Tuple
import numpy as np
import random
import json


class SA(Optimizer):
    """
    Simulated Annealing (SA) optimizer for fitting LPPL parameters.

    This class uses the Simulated Annealing (SA) algorithm to find the optimal parameters
    for the LPPL (Log-Periodic Power Law) model. The SA algorithm is a probabilistic technique
    for finding the global minimum of a function, specifically designed for optimization
    problems like fitting model parameters to time series data.

    The optimizer starts with a random solution and iteratively improves it by exploring
    candidate solutions. The acceptance of a candidate solution depends on its fitness,
    with a probability of acceptance that decreases as the algorithm progresses, controlled
    by the cooling rate.

    Attributes
    ----------
    frequency : str
        The frequency of the time series, must be one of {"daily", "weekly", "monthly"}.
    PARAM_BOUNDS : dict
        Bounds for the LPPL model parameters (t_c, omega, phi, alpha), loaded from a JSON config file.
    MAX_ITER : int
        The maximum number of iterations for the SA algorithm.
    INITIAL_TEMP : float
        The initial temperature for the SA algorithm.
    COOLING_RATE : float
        The rate at which the temperature decreases during the algorithm.
    """

    def __init__(self) -> None:
        """
        Initialize the SA optimizer with the specified frequency and load configuration parameters.
        """

        # Load optimization parameters from a JSON configuration file
        with open("params/params_sa.json", "r") as f:
            params = json.load(f)

        self.PARAM_BOUNDS = params["PARAM_BOUNDS"]
        self.MAX_ITER = params["MAX_ITER"]
        self.INITIAL_TEMP = params["INITIAL_TEMP"]
        self.COOLING_RATE = params["COOLING_RATE"]

    def fit(self, start: int, end: int, data: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Fit the LPPL model parameters to the data using Simulated Annealing (SA) optimization.

        The algorithm iteratively explores new candidate solutions, accepting them based on a
        probabilistic acceptance criterion that depends on the difference in fitness and the
        current temperature. The temperature decreases over time according to the cooling rate.

        Parameters
        ----------
        start : int
            The start index of the subinterval of the data to be used for optimization.
        end : int
            The end index of the subinterval of the data to be used for optimization.
        data : np.ndarray
            A 2D array of shape (J, 2), where:
                - Column 0 contains time values.
                - Column 1 contains observed values (e.g., stock prices or other measurements).

        Returns
        -------
        Tuple[float, np.ndarray]
            A tuple containing:
            - Best fitness value (RSS) as a float, representing the residual sum of squares
              between the model's predictions and the observed data.
            - Best solution as a 1D numpy array containing the optimal parameters for the LPPL model
              (t_c, alpha, omega, phi).
        """
        param_bounds = self.convert_param_bounds(end)
        self.fitness_history = []
        
        # Initialize the current solution randomly within parameter bounds
        current_solution = [np.random.uniform(low, high) for (low, high) in param_bounds]
        best_solution = current_solution[:]
        current_fitness = LPPL.numba_RSS(current_solution, data)
        best_fitness = current_fitness
        self.fitness_history.append(current_fitness)

        # Temperature initialization
        temperature = self.INITIAL_TEMP
        candidate_solution = np.empty_like(current_solution)

        current = 0
        while current <= self.MAX_ITER:
            # Generate a new candidate solution by sampling within parameter bounds
            for i in range(len(current_solution)):
                low, high = param_bounds[i]
                candidate_solution[i] = np.random.uniform(low, high)

            # Evaluate the fitness of the candidate solution
            candidate_fitness = LPPL.numba_RSS(candidate_solution, data)

            # Acceptance probability: Accept better solutions directly, or accept worse ones probabilistically
            if candidate_fitness < current_fitness:
                accept = True
            else:
                # As the temperature decreases, the probability of accepting worse solutions decreases
                delta = candidate_fitness - current_fitness
                accept = random.random() < np.exp(-delta / temperature)

            # If the candidate solution is accepted, update the current solution
            if accept:
                current_solution = candidate_solution
                current_fitness = candidate_fitness
                self.fitness_history.append(current_fitness)

            # Track the best solution found so far
            if current_fitness < best_fitness:
                best_solution = current_solution
                best_fitness = current_fitness
                current = 0

            current += 1
            # Cooling schedule: Gradually reduce the temperature
            temperature *= self.COOLING_RATE

        return best_fitness, best_solution