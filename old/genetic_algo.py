from dataclasses import dataclass
import random
from datetime import datetime
import pandas as pd
import numpy as np
from old.population import Population
import plotly.express as px

@dataclass
class GeneticAlgorithm:

    NUMBER_POPULATION : int = 10
    POPULATION_SIZE : int = 100
    MAX_GENERATION : int = 150
    STOP_GEN : int = 25
    SELECTION_SIZE : int = 4

    MUTATION_PROBA : float = 0.05
    CROSSOVER_PROBA : float = 0.80
    MUTATION_SCALE : float = 0.10

    def run(self, data : pd.DataFrame):

        populations = []
        min_fitness_history = {pop_nb: [] for pop_nb in range(self.NUMBER_POPULATION)}

        generation = 0
        stop_gen = 0
        previous_global_min = np.inf
        while generation <= self.MAX_GENERATION and stop_gen <= self.STOP_GEN:

            print("-----------------------------------------------------------")
            min_fitness_generation = []
            max_fitness_generation = []

            for pop_nb in range(self.NUMBER_POPULATION):

                # Initialization of all populations
                if generation == 0 :

                    pop = Population(POPULATION_SIZE=self.POPULATION_SIZE,
                                    MUTATION_PROBA=self.MUTATION_PROBA,
                                    CROSSOVER_PROBA=self.CROSSOVER_PROBA,
                                    MUTATION_SCALE=self.MUTATION_SCALE,
                                    SELECTION_SIZE=self.SELECTION_SIZE,
                                    data=data)
                    pop.initialize()
                    populations.append(pop)

                # Run the genetic algorithm iteration
                else :

                    pop = populations[pop_nb]

                    # Selection
                    selected_population = pop.selection()
                    # Crossover
                    pop.crossover(selected_population)
                    # Mutation
                    pop.mutation()

                    populations[pop_nb] = pop

                    # Get the minimum fitness of the population
                    min_fitness_individual, min_index = pop.get_min_fitness()
                    min_fitness_generation.append(min_index)

                    # Get the maximum fitness of the population
                    max_fitness_individual, max_index = pop.get_max_fitness()
                    max_fitness_generation.append(max_index)

                    print(f"Generation {generation}, Population {pop_nb} : min fitness = {min_fitness_individual.fitness(data)}")

                    # Store the minimum fitness for plotting
                    min_fitness_value = min_fitness_individual.fitness(data)
                    min_fitness_history[pop_nb].append(min_fitness_value)


            if generation > 0:

                # Migration
                for pop_nb in range(self.NUMBER_POPULATION):
                    # For the first population, the min fitness individual comes from the last population
                    if pop_nb == 0:
                        pop_max_fitness = max_fitness_generation[0]
                        last_pop_min_fitness = min_fitness_generation[-1]
                        populations[0].individuals[pop_max_fitness] = populations[-1].individuals[last_pop_min_fitness]
                    else:
                        pop_max_fitness = max_fitness_generation[pop_nb]
                        previous_pop_min_fitness = min_fitness_generation[pop_nb-1]
                        populations[pop_nb].individuals[pop_max_fitness] = populations[pop_nb-1].individuals[previous_pop_min_fitness]

                # Global minimum fitness
                min_pop = [populations[pop_nb].individuals[min_fitness_generation[pop_nb]] for pop_nb in range(self.NUMBER_POPULATION)]
                global_min_fitness = min([ind.fitness(data) for ind in min_pop])
                if global_min_fitness < previous_global_min:
                    stop_gen = 0
                    previous_global_min = global_min_fitness
                    for ind in min_pop:
                        if ind.fitness(data) == global_min_fitness:
                            print(f"****** - {ind.params} - ******")
                else:
                    stop_gen += 1

            generation += 1

        # Plot the minimum fitness convergence
        fitness_df = pd.DataFrame({
            f"Population {pop_nb}": history
            for pop_nb, history in min_fitness_history.items()
        })
        fitness_df["Generation"] = fitness_df.index
        fitness_df = pd.melt(fitness_df, id_vars="Generation", var_name="Population", value_name="Minimum Fitness (RSS)")

        fig = px.line(fitness_df, x="Generation", y="Minimum Fitness (RSS)", color="Population", title="Convergence of Genetic Algorithm")
        fig.show()