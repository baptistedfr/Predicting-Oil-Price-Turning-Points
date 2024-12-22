import pandas as pd
from scipy.optimize import minimize
import numpy as np
import random


class LPPLGeneticAlgorithm:
    def __init__(self, n_pop=100, num_populations=10, max_generations=500, stop_gen=50,
                 selection_probability=0.9, crossover_probability_range=(0.7, 0.9),
                 mutation_probability_range=(0.001, 0.05)):
        self.n_pop = n_pop
        self.num_populations = num_populations
        self.max_generations = max_generations
        self.stop_gen = stop_gen
        self.selection_probability = selection_probability
        self.crossover_probability_range = crossover_probability_range
        self.mutation_probability_range = mutation_probability_range

        self.bounds = [
            (1, 2520),  # tc
            (0.1, 0.9),  # alpha
            (0, 40),  # omega
            (0, 2 * np.pi)  # phi
        ]

    def lppl(self, t, A, B, tc, alpha, C, omega, phi):
        return A + B * (tc - t) ** alpha + C * (tc - t) ** alpha * np.cos(omega * np.log(tc - t) + phi)

    def fitness_function(self, params, t, P_actual):
        A, B, C, tc, alpha, omega, phi = params
        try:
            P_predicted = self.lppl(t, A, B, tc, alpha, C, omega, phi)
            return float(np.sum((P_actual - P_predicted) ** 2))
        except Exception:
            return 1e10  # High error for invalid parameters

    def lomb_periodogram(self, t, x, freqs, p=0.95):
        """
        Perform Lomb periodogram analysis for the given time series x at times t with a set of frequencies.

        Parameters:
        - t: Time points (array)
        - x: Observed values (array)
        - freqs: Frequency series to test (array)
        - p: Statistical significance level (default 0.95)

        Returns:
        - P: Power spectral density for each frequency in freqs
        - valid_freq: The frequency with the highest valid power
        """
        J = len(t)
        mean_x = np.mean(x)
        s2 = np.var(x, ddof=1)  # variance of x
        P = []

        # Compute the Lomb periodogram for each frequency
        for f in freqs:
            # Calculate time offset t
            sin_term = np.sum(np.sin(2 * np.pi * f * (t - t.mean())))
            cos_term = np.sum(np.cos(2 * np.pi * f * (t - t.mean())))
            t_offset = (1 / (4 * np.pi * f)) * np.arctan(sin_term / cos_term)

            # Calculate power spectral density P(f)
            cos_term = np.sum((x - mean_x) * np.cos(2 * np.pi * f * (t - t_offset)))
            sin_term = np.sum((x - mean_x) * np.sin(2 * np.pi * f * (t - t_offset)))
            power = (cos_term ** 2 + sin_term ** 2) / (J * s2)
            P.append(power)

        # Convert to array for convenience
        P = np.array(P)

        # Calculate critical value based on statistical significance level p
        M = len(freqs)
        critical_value = -np.log(1 - (1 - p) ** (1 / M))

        # Find valid frequencies based on power being greater than the critical value
        valid_freqs = freqs[P > critical_value]

        # If no valid frequencies, return None
        if len(valid_freqs) == 0:
            return None, None

        # Select the frequency with the maximum power
        max_power_idx = np.argmax(P)
        valid_freq = freqs[max_power_idx]

        return P, valid_freq

    def fit_lppl(self, data, params, P_actual):
        y = data.iloc[:, 1].values
        t = data.iloc[:, 0].values

        tc, alpha, omega, phi = params

        dt = tc - t
        if np.any(dt <= 0):
            return np.inf

        f = dt ** alpha
        g = f * np.cos(omega * np.log(dt) + phi)

        # Construction de la matrice de régression linéaire
        V = np.column_stack((np.ones_like(f), f, g))

        try:
            # Résolution du système linéaire pour trouver A, B, C
            params_lin = np.linalg.inv(V.T @ V) @ (V.T @ y)
            A, B, C = params_lin[0], params_lin[1], params_lin[2]
        except np.linalg.LinAlgError:  # Si la matrice est singulière
            return np.inf

        # Reconstruction des paramètres complets
        all_params = [float(A), float(B), float(C)] + [float(p) for p in params]

        # Évaluation de la fonction d'erreur (RSS)
        try:
            rss = self.fitness_function(all_params, t, P_actual)
            return rss
        except Exception as e:
            return np.inf

    def generate_individual(self):
        return [np.random.uniform(low, high) for (low, high) in self.bounds]

    def selection(self, population, fitness):
        """
        Perform tournament selection.
        """
        selected = []
        for _ in range(len(population)):
            candidates = random.sample(range(len(population)), k=2)
            if fitness[candidates[0]] < fitness[candidates[1]]:
                selected.append(population[candidates[0]])
            else:
                selected.append(population[candidates[1]])
        return selected

    def generate_probabilities(self):
        crossover_probability = np.random.uniform(*self.crossover_probability_range)
        mutation_probability = np.random.uniform(*self.mutation_probability_range)
        return crossover_probability, mutation_probability

    def crossover(self, parents, crossover_prob):
        """
        Perform uniform crossover between two parents.
        """
        child = parents[0][:]
        for i in range(len(child)):
            if random.random() < crossover_prob:
                child[i] = parents[1][i]
        return child

    def mutate(self, individual, mutation_prob, bounds):
        """
        Perform random mutation within bounds for an individual.
        """
        for i in range(len(individual)):
            if random.random() < mutation_prob:
                individual[i] = np.random.uniform(bounds[i][0], bounds[i][1])
        return individual

    def immigration_operation(self, populations, fitness_values):
        """
        Perform simple immigration by swapping a portion of the population between neighboring populations.
        """
        for m in range(len(populations) - 1):
            f = fitness_values[m]
            best_idx = np.argmin(f)
            # Copy best chrom from pop m
            best_chrom = populations[m][best_idx]

            f_next = fitness_values[m + 1]
            worst_idx = np.argmax(f_next)
            # Overwrite the worst in pop m+1
            populations[m + 1][worst_idx] = best_chrom
        return populations

    def run(self, data):
        t_all = np.arange(len(data))
        prices = data["Price"].values
        delta = max((len(t_all) * 0.75 / 15, 15))
        subintervals = [(int(i * delta), int((i + 1) * delta)) for i in range(4)]
        all_best_individuals = []
        all_best_fitness = []

        for interval in subintervals:
            start, end = interval
            t_segment = t_all[start:end]
            P_segment = prices[start:end]
            sub_data = pd.concat([pd.Series(t_segment), pd.Series(P_segment)], axis=1)

            populations = [[self.generate_individual() for _ in range(self.n_pop)] for _ in range(self.num_populations)]

            gen0 = 0
            best_fitness = float("inf")
            best_individual = None

            for gen in range(self.max_generations):
                print(f"Generation {gen}")
                new_populations = []
                fitness_values = np.array([
                    [self.fit_lppl(sub_data, ind, P_segment) for ind in pop]
                    for pop in populations
                ])

                for m, fitness in enumerate(fitness_values):
                    selected = self.selection(populations[m], fitness)
                    crossover_prob, mutation_prob = self.generate_probabilities()
                    offspring = [
                        self.mutate(
                            self.crossover(random.sample(selected, 2), crossover_prob),
                            mutation_prob,
                            self.bounds
                        ) for _ in range(self.n_pop)
                    ]
                    new_populations.append(offspring)

                populations = self.immigration_operation(new_populations, fitness_values)

                # Check the best individual in the current generation
                for pop_idx, fitness in enumerate(fitness_values):
                    min_idx = np.argmin(fitness)
                    if fitness[min_idx] < best_fitness:
                        best_fitness = fitness[min_idx]
                        best_individual = populations[pop_idx][min_idx]
                        gen0 = 0
                    else:
                        gen0 += 1
                print(gen0)

                if gen0 >= self.stop_gen:
                    break

            all_best_individuals.append(best_individual)
            all_best_fitness.append(best_fitness)

            # Apply Lomb periodogram to validate the predicted turning points
            P, valid_freq = self.lomb_periodogram(t_segment, P_segment, np.linspace(0.1, 1.0, 100))
            if valid_freq is not None:
                print(f"Valid frequency detected: {valid_freq}")
            else:
                print("No valid frequency found for this segment")

        return all_best_individuals, all_best_fitness


if __name__ == "__main__":
    data = pd.read_csv("WTI_Spot_Price_daily.csv", sep=";")
    data = data.iloc[4:, :]
    data[['Date', 'Price']] = data['Cushing OK WTI Spot Price FOB'].str.split(',', expand=True)
    data['Date'] = pd.to_datetime(data['Date'])
    data['Price'] = pd.to_numeric(data['Price'])
    data.set_index('Date', inplace=True)
    data.drop(columns=['Cushing OK WTI Spot Price FOB'], inplace=True)
    data.sort_index(inplace=True)

    test_data = data.loc["2004-01-01":"2008-01-01"]

    ga = LPPLGeneticAlgorithm()
    best_individuals, best_fitness = ga.run(test_data)

    print("Best Individuals:", best_individuals)
    print("Best Fitness:", best_fitness)
