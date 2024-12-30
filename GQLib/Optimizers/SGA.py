from .abstract_optimizers import GeneticAlgorithm
from typing import Tuple
import numpy as np
import json


class SGA(GeneticAlgorithm):
    """
    Standard Genetic Algorithm (SGA) for optimizing LPPL parameters.

    This optimizer evolves a single population through selection, crossover, mutation to minimize the Residual Sum of Squares (RSS).
    """

    def __init__(self) -> None:
        """
        Initialize the SGA optimizer.
        """

        # Load optimization parameters from a JSON configuration file

        self.POPULATION_SIZE = None
        self.MAX_GEN = None
        self.STOP_GEN = None

    def fit(self, start: int, end: int, data: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Fit LPPL parameters using the SGA optimizer.

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
        self.fitness_history = []

        # Generate random probabilities for crossover and mutation
        crossover_prob = np.random.uniform(0.001, 0.05)
        mutation_prob = np.random.uniform(0.001, 0.05)

        # Initialize populations for all subintervals
        population = self.initialize_population(param_bounds, self.POPULATION_SIZE)

        # Compute initial fitness values
        fitness = self.calculate_fitness(population, data)

        # Determine initial best individual
        bestObjV = np.min(fitness)
        bestChrom = population[np.argmin(fitness)]
        self.fitness_history.append(bestObjV)

        # Initialize loop counters
        gen = 1
        gen0 = 0

        # MPGA Evolution Loop
        while gen0 < self.STOP_GEN and gen <= self.MAX_GEN:

            # Perform selection, crossover and mutation
            selected = self.selection(population, fitness)
            offspring = self.crossover(selected, crossover_prob)
            mutated = self.mutate(offspring, mutation_prob, param_bounds)

            # Update population
            population = mutated

            # Recompute fitness values
            fitness = self.calculate_fitness(population, data)
            
            # Determine best individual
            newbestObjV = np.min(fitness)
            newbestChrom = population[np.argmin(fitness)]
            self.fitness_history.append(newbestObjV)

            # Update best solution
            if newbestObjV < bestObjV:
                bestObjV = newbestObjV
                bestChrom = newbestChrom
                gen0 = 0
            else:
                gen0 += 1
            
            gen += 1

        return bestObjV, bestChrom
