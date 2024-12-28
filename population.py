import random
from dataclasses import dataclass
from operator import index

from individual import Individual
import numpy as np
import pandas as pd
import math

@dataclass
class Population:

    POPULATION_SIZE : int
    MUTATION_PROBA : float
    CROSSOVER_PROBA : float
    MUTATION_SCALE : float
    SELECTION_SIZE : int

    data: pd.DataFrame
    individuals : list[Individual] = None

    def initialize(self):
        """
        Create the initial population with predetermined parameters bounds
        """
        max_day = max(self.data["Days"].values)
        bounds = [(max_day + 1, max_day + 356 * 10),  # tc
                  (0, 40),  # omega
                  (0, 2 * np.pi),  # phi
                  (0.1, 0.9)]  # alpha

        self.individuals = []
        for _ in range(self.POPULATION_SIZE):
            new_params = [np.random.uniform(low=b[0], high=b[1]) for b in bounds]
            self.individuals.append(Individual(params=new_params))

    def fitness(self):
        """
        Compute the fitness of the population
        """
        individual_fitness = []
        for individual in self.individuals:
            individual_fitness.append(individual.fitness(self.data))

        return individual_fitness

    def get_min_fitness(self) -> (Individual, int):
        """
        Return the minimum fitness individual
        """
        fitness = self.fitness()
        min_fitness_index = fitness.index(min(fitness))
        best_individual = self.individuals[min_fitness_index]

        return best_individual, min_fitness_index

    def get_max_fitness(self) -> (Individual, int):
        """
        Return the maximum fitness individual
        """
        fitness = self.fitness()
        max_fitness_index = fitness.index(max(fitness))
        worst_individual = self.individuals[max_fitness_index]

        return worst_individual, max_fitness_index

    def selection(self) -> list[Individual]:
        """
        Select the best individuals from the population by 'tournament selection' method to prepare the next generation
        X individuals are selected among the population; for each subpopulation, we select the minimum fitness
        """
        nb_stack = math.floor(self.POPULATION_SIZE / self.SELECTION_SIZE)

        new_population = []
        for _ in range(nb_stack):
            selection: list[Individual] = random.choices(self.individuals, k=self.SELECTION_SIZE)
            selection_fitness = [s.fitness(self.data) for s in selection]
            min_fitness_index = selection_fitness.index(min(selection_fitness))
            new_population.append(selection[min_fitness_index])

        return new_population

    def mutation(self):
        """
        Perform mutation for each individual according to mutation probabilities
        """
        for individual in self.individuals:
            if random.random() < self.MUTATION_PROBA:
                individual.mutation(mutation_scale=self.MUTATION_SCALE)

    def crossover(self, selected_population : list[Individual]):
        """
        Perform crossovers between pairs of individuals randomly picked among the population
        """
        new_population = []
        while len(new_population) < self.POPULATION_SIZE:

            parents: list[Individual] = random.choices(selected_population, k=2)
            if random.random() < self.CROSSOVER_PROBA:
                child1, child2 = parents[0].crossover(other_parent=parents[1])
                new_population.extend([child1, child2])
            else:
                new_population.extend(parents)

        self.individuals = new_population
