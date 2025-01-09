import numpy as np
from GQLib.Models import LPPL, LPPLS
from typing import Tuple
from .abstract_optimizer import Optimizer


class TABU(Optimizer):
    """
    Tabu Search optimizer for optimizing LPPL parameters.

    Tabu Search is a metaheuristic optimization algorithm that explores the search space by iteratively moving to the best neighboring solution,
    while avoiding cycles or revisiting recently explored solutions by maintaining a tabu list.
    """

    def __init__(self, lppl_model: 'LPPL | LPPLS' = LPPL) -> None:
        """
        Initialize the Tabu Search optimizer.

        Parameters
        ----------
        lppl_model : LPPL | LPPLS
            The model being optimized (LPPL or LPPLS).
        max_iterations : int
            Maximum number of iterations to run the optimizer.
        tabu_tenure : int
            Number of iterations a move remains in the tabu list.
        neighborhood_size : int
            Number of neighboring solutions to explore in each iteration.
        """
        self.lppl_model = lppl_model
        self.MAX_ITERATION = None
        self.TABU_TENURE = None
        self.NEIGHBORHOOD_SIZE = None
        self.tabu_list = []

    def fit(self, start: int, end: int, data: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Fit LPPL parameters using the Tabu Search optimizer.

        Parameters
        ----------
        start : int
            The start index of the subinterval.
        end : int
            The end index of the subinterval.
        data : np.ndarray
            A 2D array of shape (J, 2), where:
                - Column 0 is time.
                - Column 1 is the observed price.

        Returns
        -------
        Tuple[float, np.ndarray]
            - Best fitness value (RSS) as a float.
            - Best parameters (t_c, alpha, omega, phi) as a 1D NumPy array.
        """
        if self.lppl_model == LPPL:
            param_bounds = self.convert_param_bounds_lppl(end)
        elif self.lppl_model == LPPLS:
            param_bounds = self.convert_param_bounds_lppls(end)
        else:
            raise ValueError("Invalid model type.")

        # Initialize the current solution randomly within bounds
        current_solution = self.initialize_solution(param_bounds)
        current_fitness = self.compute_fitness(current_solution, data)

        # Set initial best solution
        best_solution = current_solution.copy()
        best_fitness = current_fitness

        # Tabu list to store recent moves
        self.tabu_list = []

        for iteration in range(self.MAX_ITERATION):
            # Generate neighborhood solutions
            neighborhood = self.generate_neighborhood(current_solution, param_bounds)

            # Evaluate neighbors and select the best non-tabu solution
            best_neighbor = None
            best_neighbor_fitness = float('inf')
            for neighbor in neighborhood:
                if neighbor.tolist() not in self.tabu_list:
                    fitness = self.compute_fitness(neighbor, data)
                    if fitness < best_neighbor_fitness:
                        best_neighbor = neighbor
                        best_neighbor_fitness = fitness

            # Update the tabu list
            self.tabu_list.append(current_solution.tolist())
            if len(self.tabu_list) > self.TABU_TENURE:
                self.tabu_list.pop(0)

            # Move to the best neighbor
            if best_neighbor is not None:
                current_solution = best_neighbor
                current_fitness = best_neighbor_fitness

                # Update global best solution
                if current_fitness < best_fitness:
                    best_solution = current_solution.copy()
                    best_fitness = current_fitness

        return best_fitness, best_solution

    def initialize_solution(self, param_bounds: np.ndarray) -> np.ndarray:
        """
        Initialize a random solution within parameter bounds.

        Parameters
        ----------
        param_bounds : np.ndarray, shape (D, 2)
            Rows correspond to each parameter [low, high].

        Returns
        -------
        np.ndarray
            A randomly initialized solution.
        """
        return np.array([np.random.uniform(low, high) for low, high in param_bounds])

    def compute_fitness(self, solution: np.ndarray, data: np.ndarray) -> float:
        """
        Compute the fitness (RSS) of a given solution.

        Parameters
        ----------
        solution : np.ndarray
            The solution (parameter set) to evaluate.
        data : np.ndarray
            Observed data.

        Returns
        -------
        float
            The fitness value (RSS).
        """
        return self.lppl_model.numba_RSS(solution, data)

    def generate_neighborhood(self, solution: np.ndarray, param_bounds: np.ndarray) -> np.ndarray:
        """
        Generate a neighborhood of solutions around the current solution.

        Parameters
        ----------
        solution : np.ndarray
            The current solution.
        param_bounds : np.ndarray
            Bounds for each parameter.

        Returns
        -------
        np.ndarray
            A list of neighboring solutions.
        """
        neighborhood = []
        for _ in range(self.NEIGHBORHOOD_SIZE):
            neighbor = solution.copy()
            for i in range(len(solution)):
                perturbation = np.random.uniform(-0.1, 0.1) * (param_bounds[i][1] - param_bounds[i][0])
                neighbor[i] = np.clip(neighbor[i] + perturbation, param_bounds[i][0], param_bounds[i][1])
            neighborhood.append(neighbor)
        return neighborhood