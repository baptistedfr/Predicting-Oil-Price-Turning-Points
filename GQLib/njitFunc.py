import numpy as np
from numba import njit

@njit
def njit_initialize_population(param_bounds: np.ndarray, population_size: int) -> np.ndarray:
    """
    Initialize a population in nopython mode with Numba.
    
    Parameters
    ----------
    param_bounds : np.ndarray, shape (D, 2)
        Rows correspond to each parameter [low, high].
        For instance, if we have 4 parameters:
            param_bounds[0] = [t_c_min, t_c_max]
            param_bounds[1] = [omega_min, omega_max]
            param_bounds[2] = [phi_min, phi_max]
            param_bounds[3] = [alpha_min, alpha_max]
    population_size : int
        Number of individuals in the population.

    Returns
    -------
    np.ndarray, shape (population_size, D)
        Each row is a chromosome with parameter values sampled from the respective [low, high].
    """
    num_params = param_bounds.shape[0]  # D
    pop = np.empty((population_size, num_params), dtype=np.float64)

    for i, bounds in enumerate(param_bounds): 
        pop[:, i] = np.random.uniform(bounds[0], bounds[1], size=population_size)
    return pop

    # for i in range(population_size):
    #     for j in range(num_params):
    #         low = param_bounds[j, 0]
    #         high = param_bounds[j, 1]
    #         pop[i, j] = np.random.uniform(low, high)
    
    # return pop

@njit
def njit_selection(population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
    """
    Tournament selection in nopython mode.
    
    Parameters
    ----------
    population : np.ndarray, shape (N, D)
        N individuals, each with D parameters.
    fitness : np.ndarray, shape (N,)
        Fitness (e.g., RSS) of each individual. Lower is better.

    Returns
    -------
    np.ndarray, shape (N, D)
        The selected population of size N, after tournament selection.
    """
    n = population.shape[0]
    num_params = population.shape[1]

    selected = np.empty((n, num_params), dtype=np.float64)

    for k in range(n):
        # Pick two random indices in [0, n)
        i, j = np.random.randint(0, n, size=2)
        
        # Select the better chromosome (lower fitness)
        if fitness[i] < fitness[j]:
            selected[k, :] = population[i, :]
        else:
            selected[k, :] = population[j, :]
        # if fitness[i] < fitness[j]:
        #     for col in range(num_params):
        #         selected[k, col] = population[i, col]
        # else:
        #     for col in range(num_params):
        #         selected[k, col] = population[j, col]

    return selected

@njit
def njit_crossover(parents: np.ndarray, prob: float) -> np.ndarray:
    """
    Single-point crossover in nopython mode.
    
    Parameters
    ----------
    parents : np.ndarray, shape (N, D)
        N individuals (rows), each with D parameters (columns).
    prob : float
        Probability of applying crossover to each pair.

    Returns
    -------
    np.ndarray, shape (N, D)
        Offspring population after crossover.
    """
    n, d = parents.shape
    offspring = parents.copy()

    # Process pairs of parents
    # for i in range(0, n, 2):
    #     if i + 1 >= n:
    #         # If there's an odd number of parents, copy the last one as is
    #         # for col in range(d):
    #         #     offspring[i, col] = parents[i, col]
    #         offspring[i, :] = parents[i, :]
    #         break

    #     # Decide whether to crossover
    #     if np.random.rand() < prob:
    #         cp = np.random.randint(1, d)  # single crossover point
    #         # Child1
    #         for col in range(cp):
    #             offspring[i, col] = parents[i, col]
    #         for col in range(cp, d):
    #             offspring[i, col] = parents[i+1, col]
    #         # Child2
    #         for col in range(cp):
    #             offspring[i+1, col] = parents[i+1, col]
    #         for col in range(cp, d):
    #             offspring[i+1, col] = parents[i, col]
    #     else:
    #         # No crossover -> copy parents as-is
    #         # for col in range(d):
    #         #     offspring[i, col] = parents[i, col]
    #         #     offspring[i+1, col] = parents[i+1, col]
    #         offspring[i, :] = parents[i, :]
    #         offspring[i+1, :] = parents[i+1, :]
    for i in range(0, n - 1, 2):
        if np.random.rand() < prob:
            cp = np.random.randint(1, d)  # Point de croisement
            offspring[i, cp:], offspring[i+1, cp:] = parents[i+1, cp:], parents[i, cp:]  # Croisement en une ligne
    return offspring

@njit
def njit_mutate(offspring: np.ndarray, prob: float, param_bounds: np.ndarray) -> np.ndarray:
    """
    Mutation operator in nopython mode.

    Parameters
    ----------
    offspring : np.ndarray, shape (N, D)
        The current population. Each row is an individual, each column is a parameter.
    prob : float
        Mutation probability for each individual.
    param_bounds : np.ndarray, shape (D, 2)
        [low, high] bounds for each of the D parameters.

    Returns
    -------
    np.ndarray, shape (N, D)
        Mutated population, in-place.
    """
    n, d = offspring.shape
    mutation_mask = np.random.rand(n) < prob
    for i in range(n):
        # if np.random.rand() < prob:
        if mutation_mask[i]:
            # pick a random parameter index
            mp = np.random.randint(d)
            low = param_bounds[mp, 0]
            high = param_bounds[mp, 1]
            offspring[i, mp] = np.random.uniform(low, high)

    return offspring

@njit
def njit_immigration_operation(populations: np.ndarray, fitness_values: np.ndarray) -> np.ndarray:
    """
    Immigration operation in nopython mode.
    The best individual of population m replaces the worst individual of population m+1.
    
    Parameters
    ----------
    populations : np.ndarray, shape (M, N, D)
        M populations, each with N individuals, each individual has D parameters.
        i.e. a 3D array [pop_index, ind_index, param_index].
    fitness_values : np.ndarray, shape (M, N)
        The fitness array corresponding to each population, shape (M, N).

    Returns
    -------
    np.ndarray, shape (M, N, D)
        Updated populations after immigration.
    """
    # We assume populations.shape[0] = fitness_values.shape[0]
    # and populations.shape[1] = fitness_values.shape[1]
    for m in range(len(populations) - 1):
        best_idx = np.argmin(fitness_values[m])
        worst_idx = np.argmax(fitness_values[m+1])
        populations[m+1][worst_idx, :] = populations[m][best_idx, :]
        # # Copy best chrom from pop m
        # best_chrom = populations[m][best_idx]

        # # Overwrite the worst in pop m+1
        # populations[m+1][worst_idx] = best_chrom
    return populations

@njit
def njit_RSS(chromosome: np.ndarray, data: np.ndarray) -> float:
    """
    Compute the residual sum of squares (RSS) for the simplified LPPL model:
        y(t) ~ A + B * (t_c - t)^alpha + C * (t_c - t)^alpha * cos(omega ln(t_c - t) + phi).

    Parameters
    ----------
    chromosome : np.ndarray, shape (4,)
        The nonlinear parameters: [t_c, alpha, omega, phi].
    data : np.ndarray, shape (J, 2)
        The time series data. data[:,0] = t, data[:,1] = y.

    Returns
    -------
    float
        The RSS value of the fit. If the linear system is non-invertible, returns np.inf.
    """
    y = data[:, 1]
    t = data[:, 0]
    t_c, alpha, omega, phi = chromosome

    dt = t_c - t
    f = dt ** alpha
    g = f * np.cos(omega * np.log(dt) + phi)

    # Build design matrix
    V = np.column_stack((np.ones_like(f), f, g))
    # Attempt to invert (V^T V)
    try:
        params = np.linalg.inv(V.T @ V) @ (V.T @ y)
        A, B, C = params[0], params[1], params[2]
    except:
        return np.inf

    predicted = A + B*f + C*g
    return np.sum((y - predicted)**2)

@njit
def njit_calculate_fitness(population: np.ndarray, data: np.ndarray) -> np.ndarray:
    """
    Calculate the fitness (RSS) for each chromosome in the population.

    Parameters
    ----------
    population : np.ndarray, shape (N, 4)
        Each row is a chromosome [t_c, alpha, omega, phi].
    data : np.ndarray, shape (J, 2)
        The time series, where data[:,0] = t, data[:,1] = y.

    Returns
    -------
    np.ndarray, shape (N,)
        The RSS of each individual in the population.
    """
    n = len(population)
    fitness = np.empty(n, dtype=np.float64)
    for i in range(n):
        fitness[i] = njit_RSS(population[i], data)
    return fitness

@njit
def njit_update_velocity(velocity, position, local_min, global_best, w, c1, c2):
    """
    Update the velocity of a particle in a swarm optimization process.

    Parameters
    ----------
    velocity : np.ndarray
        Current velocity of the particle.
    position : np.ndarray
        Current position of the particle.
    local_min : np.ndarray
        Best position found by the particle.
    global_best : np.ndarray
        Best position found by the swarm.
    w : float
        Inertia weight.
    c1 : float
        Cognitive coefficient.
    c2 : float
        Social coefficient.

    Returns
    -------
    np.ndarray
        Updated velocity of the particle.
    """
    r1, r2 = np.random.uniform(0, 1, size=2)  # Random factors
    new_velocity = w * velocity + r1 * c1 * (local_min - position) + r2 * c2 * (global_best - position)
    return new_velocity

@njit
def njit_update_position(position, velocity):
    return position + velocity

