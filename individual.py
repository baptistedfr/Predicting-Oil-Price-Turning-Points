from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class Individual:

    params : list[float] # tc - omega - phi - alpha

    A : float = 1.0
    B : float = 1.0
    C : float = 1.0

    def lppl(self, t):
        if self.params[0] <= t:
            return np.inf,
        else:
            return (self.A + self.B * (self.params[0] - t) ** self.params[3] +
                    self.C * (self.params[0] - t) ** self.params[3] * np.cos(self.params[1] * np.log(self.params[0] - t) + self.params[2]))

    def fitness(self, data : pd.DataFrame):
        """
        Compute the fitness of the given individual
        """
        y_real = np.array(data["Spot"].values)
        temp = [self.lppl(t) for t in data["Days"].values]
        y_prediction = np.asarray(temp)

        return np.sum((y_real - y_prediction) ** 2)

    def mutation(self, mutation_scale : float):
        """
        If a mutation is triggered, choose a parameter randomly and perform a slight modification

        :param mutation_scale: scale of the parameter modification, ex: 1% -> mutation between -1% & 1%
        """
        random_param = np.random.randint(low=0, high=len(self.params))
        chromosome = self.params[random_param]
        mutation_value = np.random.uniform(low=chromosome * -mutation_scale, high=chromosome * mutation_scale)
        self.params[random_param] = chromosome + mutation_value

    def crossover(self, other_parent : "Individual"):
        """
        Perform a crossover between two parents to get two new children by concatenating portions of each parent data
        ex : parent1 = [a, b, c] & parent2 = [d, e, f] -> child1 = [a, e, f] & child2 = [d, b, c]

        :param other_parent: other instance of Individual to perform the crossover with
        """
        crossover_chromosome = np.random.randint(low=1, high=len(self.params)-1)
        params_1 = other_parent.params[:crossover_chromosome] + self.params[crossover_chromosome:]
        params_2 = self.params[:crossover_chromosome] + other_parent.params[crossover_chromosome:]

        return Individual(params=params_1), Individual(params=params_2)
