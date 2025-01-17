from typing import Tuple
import numpy as np
from ..Models import LPPL, LPPLS
from .abstract_optimizer import GeneticAlgorithm
from ..Models import LPPL, LPPLS
from GQLib.njitFunc import (

    njit_immigration_operation
)

class MPGA(GeneticAlgorithm):
    """
    Multi-Population Genetic Algorithm (MPGA) for optimizing LPPL parameters.

    This optimizer evolves multiple populations through selection, crossover, mutation,
    and immigration operations to minimize the Residual Sum of Squares (RSS).
    """

    def __init__(self, lppl_model: 'LPPL | LPPLS' = LPPL) -> None:
        """
        Initialize the MPGA optimizer.
        lppl_model : 'LPPL | LPPLS'
            Log Periodic Power Law Model to optimized
        """
        self.lppl_model = lppl_model
        self.NUM_POPULATIONS = None
        self.POPULATION_SIZE = None
        self.MAX_GEN = None
        self.STOP_GEN = None

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
        if self.lppl_model == LPPL:
            param_bounds = self.convert_param_bounds_lppl(end)
        elif self.lppl_model == LPPLS:
            param_bounds = self.convert_param_bounds_lppls(end)
        self.fitness_history = [[] for _ in range(self.NUM_POPULATIONS)]
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
            fit = self.calculate_fitness(populations[m], data, self.lppl_model)
            fitness_values.append(fit)
            local_min = np.min(fit)

            if local_min < bestObjV:
                bestObjV = local_min
                bestChrom = populations[m][np.argmin(fit)]
        self.fitness_history[m].append(bestObjV)

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
                fit = np.array([self.lppl_model.numba_RSS(ch, data) for ch in populations[m]])
                fitness_values.append(fit)

            # Check for global best solution
            newbestObjV = np.inf
            newbestChrom = None

            for m in range(self.NUM_POPULATIONS):
                local_min = np.min(fitness_values[m])

                if local_min < newbestObjV:
                    newbestObjV = local_min
                    newbestChrom = populations[m][np.argmin(fitness_values[m])]
            self.fitness_history[m].append(newbestObjV)

            # Update counters based on improvement
            if newbestObjV < bestObjV:
                bestObjV = newbestObjV
                bestChrom = newbestChrom
                gen0 = 0
            else:
                gen0 += 1

            gen += 1

        return bestObjV, bestChrom

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

    