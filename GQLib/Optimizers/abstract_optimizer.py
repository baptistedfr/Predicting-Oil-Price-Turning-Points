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
import plotly.graph_objects as go
import json
from ..Models import LPPL, LPPLS
class Optimizer(ABC):
    """
    Abstract base class for optimizers.

    Defines the interface for any optimizer used to fit LPPL parameters.
    """

    PARAM_BOUNDS = None

    @abstractmethod
    def __init__(self) -> None:
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
        self.fitness_history = []
        pass

    def convert_param_bounds_lppl(self, end: float) -> np.ndarray:
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
    
    def visualize_convergence(self):
        """
        Plot the fitness history stored in Optimizer instance to show the convergence of the algorithm.
        """
        if len(self.fitness_history) == 0:
            raise Exception("Fitness history is empty !")

        # If the algorithm has a single population
        if all(isinstance(f, float) for f in self.fitness_history):
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(len(self.fitness_history))),
                y=self.fitness_history,
                mode='lines',
                name='RSS'))   
        # If the algorithm has multiple populations
        elif all(isinstance(f, list) for f in self.fitness_history):
            fig = go.Figure()
            for series in enumerate(self.fitness_history):
                if not len(series):
                    fig.add_trace(go.Scatter(
                        x=list(range(len(series))),
                        y=series,
                        mode='lines',
                        name='RSS'))
        else:
            raise Exception("Invalid fitness history type !")

        fig.update_layout(
        title=f"Convergence of the algorithm {self.__class__.__name__}",
        xaxis_title="Iteration",
        yaxis_title="RSS")

        fig.show()

    def configure_params_from_frequency(self, frequency: str, optimizer_name : str):
        """
        Configure the optimizer's parameters based on the given frequency

        Parameters
        ----------
        frequency : str
            The frequency of the analysis, e.g., 'daily', 'weekly', or 'monthly'.
        optimizer_name : str
            The name of the optimizer being used to fetch specific parameter bounds.

        """
        try:
            with open(f"params/params_{optimizer_name.lower()}.json", "r") as f:
                params = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file for {optimizer_name} not found.")

        # Charger les paramètres liés à la fréquence
        freq_key = f"{frequency.upper()}_PARAM_BOUNDS"
        if freq_key in params:
            self.PARAM_BOUNDS = params[freq_key]

        # Créer dynamiquement des attributs pour les autres paramètres globaux
        for key, value in params.items():
            if key != freq_key:
                setattr(self, key, value)

    def convert_param_bounds_lppls(self, end: float) -> np.ndarray:
        """
        Convert parameter bounds to a NumPy array format.

        Parameters
        ----------
        end : float
            The end time of the subinterval.

        Returns
        -------
        np.ndarray
            A 2D array of shape (3, 2) representing the bounds for each parameter.
        """
        return np.array([
            [self.PARAM_BOUNDS["t_c"][0] + end,    self.PARAM_BOUNDS["t_c"][1] + end],
            [self.PARAM_BOUNDS["omega"][0],       self.PARAM_BOUNDS["omega"][1]],
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

    def calculate_fitness(self, population: np.ndarray, data: np.ndarray, lppl_model: 'LPPL | LPPLS' = LPPL ) -> np.ndarray:
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
        return njit_calculate_fitness(population, data, lppl_model)
    
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





    

    

   


