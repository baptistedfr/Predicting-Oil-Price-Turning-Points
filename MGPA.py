import pandas as pd
from scipy.optimize import minimize
import numpy as np
import random


# LPPL Function Definition
def lppl(t, A, B, tc, alpha, C, omega, phi):
    return A + B * (tc - t) ** alpha + C * (tc - t) ** alpha * np.cos(omega * np.log(tc - t) + phi)


# Step 2: Set time variables
time_start = 0
time_end = 1008 # Soit 252j * 4 pour 4 ans de données


n_pop = 100
num_populations = 10
max_generations = 500
stop_gen = 50
selection_probability = 0.9
crossover_probability_range = (0.7, 0.9)
mutation_probability_range = (0.001, 0.05)

def fit_lppl(params, t, P_actual):
    # Je fixe les params linéaires à 1
    A, B, C = 1, 1, 1
    tc, alpha, omega, phi = params

    all_params = [A, B, C] + list(params)
    bounds = [
        (1,1), # A
        (1,1),  # B
        (1,1),  # C
        (1, 2520),  # tc
        (0.1, 0.9),  # alpha
        (0, 40),  # omega
        (0, 2 * np.pi)  # phi
    ]
    result = minimize(fitness_function, all_params, args=(t, P_actual), bounds=bounds, method="L-BFGS-B")
    return result.x if result.success else None

# Minimiez RSS function for LPPL
def fitness_function(all_params, t, P_actual):
    A, B, C, tc, alpha, omega, phi = all_params
    try:
        P_predicted = lppl(t, A, B, tc, alpha, C, omega, phi)
        return np.sum((P_actual - P_predicted) ** 2)
    except Exception:
        return 1e10  # High error for invalid params



# Step 9-13: Subinterval setup, surement faux
delta = max((time_end - time_start) * 0.75 / 15, 15)
subintervals = [(time_start + i * delta, time_start + (i + 1) * delta) for i in range(4)]
# subintervals = np.arange(time_start, time_end, delta)

bounds = [
    (1, 2520), # tc
    (0.1, 0.9), # alpha
    (0, 40), # omega
    (0, 2 * np.pi) # phi
]

# Je genere mes 100 populations (100 choix de params en gros)
def generate_individual(bounds):
    return [np.random.uniform(low, high) for (low, high) in bounds]

# Step 14: Generate crossover and mutation probabilities
def generate_probabilities():
    crossover_probability = np.random.uniform(*crossover_probability_range)
    mutation_probability = np.random.uniform(*mutation_probability_range)
    return crossover_probability, mutation_probability


# Step 15-26: Genetic Algorithm implementation
def genetic_algorithm(objective, data, n_pop, mutation_probability, crossover_probability):
    t_all = np.array(range(len(data)))
    prices = data["Price"].values
    turning_points = []

    for interval in subintervals:
        start = int(interval[0])
        end = int(interval[1])

        # Découper les segments
        t_segment = t_all[start:end]
        P_segment = prices[start:end]


        pop = [generate_individual(bounds) for _ in range(n_pop)]

        for gen in range(max_generations):
            i = 0
            fitness_values = [fit_lppl(pop[i], t_segment, P_segment) for i in range(len(pop))]
            best_fitness = fitness_values[:int(len(pop) * selection_probability)]
            best_individual = [pop[i] for i in range(len(best_fitness))]

            # Stop condition
            if gen >= stop_gen:
                break

            new_population = []
            crossover_probability, mutation_probability = generate_probabilities()
########################################################################################################################
####################################### VALID JUSQUE ICI ###############################################################
########################################################################################################################
            while len(new_population) < len(pop) - len(best_individual):
                # Selection
                parents = random.choices(best_individual, k=2)

            # Crossover
                if random.random() < crossover_probability:
                    crossover_point = random.randint(1, len(parents[0]) - 1)
                    child1 = np.concatenate((parents[0][:crossover_point], parents[1][crossover_point:]))
                    child2 = np.concatenate((parents[1][:crossover_point], parents[0][crossover_point:]))
                else:
                    child1, child2 = parents

                # Mutation
                for child in [child1, child2]:
                    if random.random() < mutation_probability:
                        mutation_point = random.randint(0, len(child) - 1)
                        child[mutation_point] = np.random.uniform(value_ranges[0], value_ranges[1])

                new_population.extend([child1, child2])

            population = new_population

    return best_individual, best_fitness

if __name__ == "__main__":
    data = pd.read_csv("WTI_Spot_Price_daily.csv", sep=";")
    data = data.iloc[4:, :]
    data[['Date', 'Price']] = data['Cushing OK WTI Spot Price FOB'].str.split(',', expand=True)
    data['Date'] = pd.to_datetime(data['Date'])

    data['Price'] = pd.to_numeric(data['Price'])

    data.set_index('Date', inplace=True)
    data = data.drop(columns = ['Cushing OK WTI Spot Price FOB'])
    data = data.sort_index()
    test_data = data.loc["2003-04-01":"2016-11-14"]
    # Execute the algorithm
    best_individual, best_fitness = genetic_algorithm(lppl, test_data, n_pop, mutation_probability=0.01, crossover_probability=0.8)
    print("Best Individual:", best_individual)
    print("Best Fitness:", best_fitness)
