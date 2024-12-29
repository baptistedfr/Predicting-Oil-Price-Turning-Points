from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
from GQLib.njitFunc import (
    njit_calculate_fitness,
    njit_selection,
    njit_crossover,
    njit_mutate,
    njit_initialize_population,
)

class Optimizer(ABC):
    """
    Abstract base class for optimizers.

    Defines the interface for any optimizer used to fit LPPL parameters.
    """

    PARAM_BOUNDS = None

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
    
    
class GeneticAlgorithm(Optimizer):
    """
    Abstract base class for genetic algorithms.

    Define the standard methods for :
        - Population initialization
        - Population fitness
        - Process of : selection, mutation and crossover
    """

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






    

    

   


