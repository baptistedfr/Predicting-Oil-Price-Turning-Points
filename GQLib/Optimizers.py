from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
from GQLib.LPPL import LPPL
import json
from GQLib.njitFunc import (
    njit_calculate_fitness,
    njit_selection,
    njit_crossover,
    njit_immigration_operation,
    njit_mutate,
    njit_initialize_population
)

class Optimizer(ABC):
    """
    Abstract base class for optimizers.

    Defines the interface for any optimizer used to fit LPPL parameters.
    """

    @abstractmethod
    def __init__(self, frequency: str) -> None:
        """
        Initialize the optimizer with a specific frequency.

        Parameters
        ----------
        frequency : str
            The frequency of the time series, must be one of {"daily", "weekly", "monthly"}.
        """
        pass

    @abstractmethod
    def fit(self, start: int, end: int, data: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Fit the model parameters to a given subinterval of the data.

        Parameters
        ----------
        start : int
            The start index of the subinterval.
        end : int
            The end index of the subinterval.
        data : np.ndarray
            A 2D array with time in the first column and observed values in the second.

        Returns
        -------
        Tuple[float, np.ndarray]
            The best fitness value (RSS) and the corresponding parameter set.
        """
        pass


class MPGA(Optimizer):
    """
    Multi-Population Genetic Algorithm (MPGA) for optimizing LPPL parameters.

    This optimizer evolves multiple populations through selection, crossover, mutation,
    and immigration operations to minimize the Residual Sum of Squares (RSS).
    """

    def __init__(self, frequency: str) -> None:
        """
        Initialize the MPGA optimizer.

        Parameters
        ----------
        frequency : str
            The frequency of the time series, must be one of {"daily", "weekly", "monthly"}.

        Raises
        ------
        ValueError
            If frequency is not one of the accepted values.
        """
        self.frequency = frequency

        # Load optimization parameters from a JSON configuration file
        with open("params.json", "r") as f:
            params = json.load(f)

        self.PARAM_BOUNDS = params["PARAM_BOUNDS"]
        self.NUM_POPULATIONS = params["NUM_POPULATIONS"]
        self.POPULATION_SIZE = params["POPULATION_SIZE"]
        self.MAX_GEN = params["MAX_GEN"]
        self.STOP_GEN = params["STOP_GEN"]

    def fit(self, start: int, end: int, data: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Fit LPPL parameters using the MPGA optimizer.

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
            - Best chromosome (parameters: t_c, alpha, omega, phi) as a 1D NumPy array.
        """
        param_bounds = self.convert_param_bounds(end)

        # Generate random probabilities for crossover and mutation
        crossover_prob = np.random.uniform(0.001, 0.05, size=self.NUM_POPULATIONS)
        mutation_prob = np.random.uniform(0.001, 0.05, size=self.NUM_POPULATIONS)

        # Initialize populations for all subintervals
        populations = [
            self.initialize_population(param_bounds, self.POPULATION_SIZE)
            for _ in range(self.NUM_POPULATIONS)
        ]

        # Compute initial fitness values
        fitness_values = []
        bestObjV = np.inf
        bestChrom = None

        for m in range(self.NUM_POPULATIONS):
            fit = self.calculate_fitness(populations[m], data)
            fitness_values.append(fit)
            local_min = np.min(fit)
            if local_min < bestObjV:
                bestObjV = local_min
                bestChrom = populations[m][np.argmin(fit)]

        # Initialize loop counters
        gen = 1
        gen0 = 0

        # MPGA Evolution Loop
        while gen0 < self.STOP_GEN and gen <= self.MAX_GEN:
            new_populations = []

            for m in range(self.NUM_POPULATIONS):
                fit = fitness_values[m]
                selected = self.selection(populations[m], fit)
                offspring = self.crossover(selected, crossover_prob[m])
                mutated = self.mutate(offspring, mutation_prob[m], param_bounds)
                new_populations.append(mutated)

            # Immigration operation
            populations = self.immigration_operation(new_populations, fitness_values)

            # Recompute fitness values
            fitness_values = []
            for m in range(self.NUM_POPULATIONS):
                fit = np.array([LPPL.numba_RSS(ch, data) for ch in populations[m]])
                fitness_values.append(fit)

            # Check for global best solution
            newbestObjV = np.inf
            newbestChrom = None
            for m in range(self.NUM_POPULATIONS):
                local_min = np.min(fitness_values[m])
                if local_min < newbestObjV:
                    newbestObjV = local_min
                    newbestChrom = populations[m][np.argmin(fitness_values[m])]

            # Update counters based on improvement
            if newbestObjV < bestObjV:
                bestObjV = newbestObjV
                bestChrom = newbestChrom
                gen0 = 0
            else:
                gen0 += 1

            gen += 1

        return bestObjV, bestChrom

    def convert_param_bounds(self, end: float) -> np.ndarray:
        """
        Convert parameter bounds to a NumPy array format.

        Parameters
        ----------
        end : float
            The end time of the subinterval.

        Returns
        -------
        np.ndarray
            A 2D array of shape (4, 2) representing the bounds for each parameter.
        """
        return np.array([
            [self.PARAM_BOUNDS["t_c"][0] + end,    self.PARAM_BOUNDS["t_c"][1] + end],
            [self.PARAM_BOUNDS["omega"][0],       self.PARAM_BOUNDS["omega"][1]],
            [self.PARAM_BOUNDS["phi"][0],         self.PARAM_BOUNDS["phi"][1]],
            [self.PARAM_BOUNDS["alpha"][0],       self.PARAM_BOUNDS["alpha"][1]]
        ], dtype=np.float64)

    def initialize_population(self, param_bounds: np.ndarray, population_size: int) -> np.ndarray:
        """
        Initialize a population of chromosomes.

        Parameters
        ----------
        param_bounds : np.ndarray
            Bounds for each parameter, shape (4, 2).
        population_size : int
            Number of individuals in the population.

        Returns
        -------
        np.ndarray
            A randomly initialized population of chromosomes.
        """
        return njit_initialize_population(param_bounds, population_size)

    def selection(self, population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        """
        Perform tournament selection.

        Parameters
        ----------
        population : np.ndarray
            Population array of shape (N, 4).
        fitness : np.ndarray
            Fitness array of shape (N,).

        Returns
        -------
        np.ndarray
            Selected individuals for the next generation.
        """
        return njit_selection(population, fitness)

    def crossover(self, parents: np.ndarray, prob: float) -> np.ndarray:
        """
        Perform single-point crossover on the population.

        Parameters
        ----------
        parents : np.ndarray
            Parent population, shape (N, 4).
        prob : float
            Probability of crossover.

        Returns
        -------
        np.ndarray
            Offspring population.
        """
        return njit_crossover(parents, prob)

    def mutate(self, offspring: np.ndarray, prob: float, param_bounds: np.ndarray) -> np.ndarray:
        """
        Apply mutation to the offspring.

        Parameters
        ----------
        offspring : np.ndarray
            Offspring population, shape (N, 4).
        prob : float
            Mutation probability.
        param_bounds : np.ndarray
            Bounds for each parameter.

        Returns
        -------
        np.ndarray
            Mutated population.
        """
        return njit_mutate(offspring, prob, param_bounds)

    def immigration_operation(self, populations: list[np.ndarray], fitness_values: list[np.ndarray]) -> list[np.ndarray]:
        """
        Perform the immigration operation between populations.

        Parameters
        ----------
        populations : list of np.ndarray
            List of populations, one per subinterval.
        fitness_values : list of np.ndarray
            List of fitness values for each population.

        Returns
        -------
        list of np.ndarray
            Updated populations after immigration.
        """
        return njit_immigration_operation(populations, fitness_values)

    def calculate_fitness(self, population: np.ndarray, data: np.ndarray) -> np.ndarray:
        """
        Calculate the RSS fitness for each individual in the population.

        Parameters
        ----------
        population : np.ndarray
            Population array, shape (N, 4).
        data : np.ndarray
            Subinterval data, shape (J, 2).

        Returns
        -------
        np.ndarray
            RSS fitness values for the population.
        """
        return njit_calculate_fitness(population, data)