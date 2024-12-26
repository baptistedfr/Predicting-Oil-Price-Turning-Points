from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
from .LPPL import LPPL
import json
from .njitFunc import (
    njit_calculate_fitness,
    njit_selection,
    njit_crossover,
    njit_immigration_operation,
    njit_mutate,
    njit_initialize_population,
    njit_update_velocity,
    njit_update_position
)
import random

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

class PSO(Optimizer):
    """
    Particule Swarm Optimisation for optimizing LPPL parameters.

    This optimizer evolves multiple swarm to minimize the Residual Sum of Squares (RSS).
    """

    def __init__(self, frequency: str) -> None:
        """
        Initialize the PSO optimizer.

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
        with open("params_pso.json", "r") as f:
            params = json.load(f)

        self.PARAM_BOUNDS = params["PARAM_BOUNDS"]
        self.NUM_PARTICULES = params["NUM_PARTICULES"]
        self.MAX_GEN = params["MAX_GEN"]

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

        # Initialize particules and calculate initial fitness values
        particules = [
            Particule(param_bounds, data)
            for _ in range(self.NUM_PARTICULES)
        ]
        # Compute global best initial fitness values
        best_particle : Particule = min(particules, key=lambda p: p.best_local_fitness)
        global_best_fitness = best_particle.best_local_fitness
        global_best_solution = best_particle.best_position

        current = 0
        while current <= self.MAX_GEN:
            '''
            Boucle principale iterrant les particules
            '''
            for m in range(self.NUM_PARTICULES):
                particules[m].update_position_fitness(global_best_solution, data)
                
            best_particle : Particule = min(particules, key=lambda p: p.best_local_fitness)
            if best_particle.best_local_fitness < global_best_fitness: #Update Global Best
                global_best_fitness = best_particle.best_local_fitness
                global_best_solution = best_particle.best_position
            current+=1
        
        return global_best_fitness, global_best_solution


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
    
class Particule():

    def __init__(self, param_bounds, data):
        """
        Initialize a particule for PSO in nopython mode with Numba.
        
        Parameters
        ----------
        param_bounds : np.ndarray, shape (D, 2)
            Rows correspond to each parameter [low, high].
            For instance, if we have 4 parameters:
                param_bounds[0] = [t_c_min, t_c_max]
                param_bounds[1] = [omega_min, omega_max]
                param_bounds[2] = [phi_min, phi_max]
                param_bounds[3] = [alpha_min, alpha_max]

        """
        num_params = param_bounds.shape[0]  # D
        self.position = np.empty(num_params, dtype=np.float64)

        for i, bounds in enumerate(param_bounds): 
            self.position[i] = np.random.uniform(bounds[0], bounds[1])
        
        self.best_position = None
        self.best_local_fitness = None
        self.compute_fitness(data)

        self.velocity = np.zeros(num_params)
        self.w = 1
        self.c1 = 1
        self.c2 = 1
    
    def compute_fitness(self, data):
        fit = LPPL.numba_RSS(self.position, data)
        if self.best_position is not None:
            if fit < self.best_local_fitness: # On actualise le minimum local
                self.best_position = self.position
                self.best_local_fitness = fit
        else: # Cas d'initialisation
            self.best_position = self.position
            self.best_local_fitness = fit

    def update_position_fitness(self, global_best, data):
        self.velocity = njit_update_velocity(self.velocity, self.position, self.best_position, global_best, self.w, self.c1, self.c2)
        self.position = njit_update_position(self.position, self.velocity)
        
        self.compute_fitness(data)
    
    
        

