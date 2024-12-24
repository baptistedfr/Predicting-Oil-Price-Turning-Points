import numpy as np
import json
from Utilities import convert_param_bounds
from njitFunc import (
    njit_calculate_fitness,
    njit_selection,
    njit_crossover,
    njit_immigration_operation,
    njit_mutate,
    njit_initialize_population,
    njit_RSS
)

class MPGA:
    """
    Multi-Population Genetic Algorithm (MPGA) class for LPPL parameter optimization.

    This class orchestrates:
      - Subinterval generation
      - Population initialization
      - Selection, crossover, mutation, and immigration operations
      - Fitness (RSS) calculation
      - Iterative evolution until convergence or maximum iteration is reached
    """

    def __init__(self, sample: np.ndarray, frequency: str = "daily") -> None:
        """
        Initialize an MPGA instance with sample data and frequency options.

        Parameters
        ----------
        sample : np.ndarray
            A 2D array of shape (N, 2), where:
                - Column 0 is time (already converted to a numeric or float scale).
                - Column 1 is the observed price or value.
            The array must be sorted in ascending time order.
        frequency : str, optional
            The frequency used for subinterval calculation. Must be one of:
            {"daily", "weekly", "monthly"}. Defaults to "daily".

        Raises
        ------
        ValueError
            If the sample is not 2-dimensional, or if the frequency is not
            one of {"daily", "weekly", "monthly"}.
        """
        # Verify if the sample is a 2D array
        if len(sample.shape) != 2:
            raise ValueError("The sample must be a 2D array.")

        self.sample = sample

        if frequency not in ["daily", "weekly", "monthly"]:
            raise ValueError("The frequency must be one of 'daily', 'weekly', 'monthly'.")
        self.frequency = frequency

        # Load parameters from JSON
        with open("params.json", "r") as f:
            params = json.load(f)

        self.PARAM_BOUNDS = params["PARAM_BOUNDS"]
        self.NUM_POPULATIONS = params["NUM_POPULATIONS"]
        self.POPULATION_SIZE = params["POPULATION_SIZE"]
        self.MAX_GEN = params["MAX_GEN"]
        self.STOP_GEN = params["STOP_GEN"]

        # Generate the subintervals
        self.generate_subintervals()

    def fit(self) -> None:
        """
        Run the MPGA on each subinterval to optimize the LPPL parameters.

        For each subinterval:
          - Initialize populations and compute initial fitness.
          - Evolve populations with selection, crossover, mutation, and immigration.
          - Track the best solution (minimum RSS) across generations.
          - Once convergence or max generation is reached, store the best solution.

        The results are appended to an internal list of best solutions per subinterval.
        Currently, the method is only constructing the results but not returning them.
        You may adapt it to store or return the results for further usage.
        """
        # Store results for each subinterval
        results = []

        # Loop over each subinterval
        for (sub_start, sub_end, sub_data) in self.subintervals:

            param_bounds = convert_param_bounds(self.PARAM_BOUNDS, sub_end)

            # Random crossover and mutation probabilities per population
            crossover_prob = np.random.uniform(0.001, 0.05, size=self.NUM_POPULATIONS)
            mutation_prob = np.random.uniform(0.001, 0.05, size=self.NUM_POPULATIONS)

            # Initialize populations
            populations = [
                self.initialize_population(param_bounds, self.POPULATION_SIZE)
                for _ in range(self.NUM_POPULATIONS)
            ]

            # Compute initial fitness
            fitness_values = []
            bestObjV = np.inf
            bestChrom = None

            for m in range(self.NUM_POPULATIONS):
                fit = self.calculate_fitness(populations[m], sub_data)
                fitness_values.append(fit)
                local_min = np.min(fit)
                if local_min < bestObjV:
                    bestObjV = local_min
                    bestChrom = populations[m][np.argmin(fit)]

            # Initialize loop counters
            gen = 1
            gen0 = 0

            # Main MPGA loop
            while gen0 < self.STOP_GEN and gen <= self.MAX_GEN:
                # Genetic operations
                new_populations = []

                for m in range(self.NUM_POPULATIONS):
                    fit = fitness_values[m]
                    # Selection
                    selected = self.selection(populations[m], fit)
                    # Crossover
                    offspring = self.crossover(selected, crossover_prob[m])
                    # Mutation
                    mutated = self.mutate(offspring, mutation_prob[m], param_bounds)
                    new_populations.append(mutated)

                # Immigration
                populations = self.immigration_operation(new_populations, fitness_values)

                # Recompute fitness after genetic operations
                fitness_values = []
                for m in range(self.NUM_POPULATIONS):
                    fit = np.array([self.RSS(ch, sub_data) for ch in populations[m]])
                    fitness_values.append(fit)

                # Find the best in the current generation
                newbestObjV = np.inf
                newbestChrom = None
                for m in range(self.NUM_POPULATIONS):
                    local_min = np.min(fitness_values[m])
                    if local_min < newbestObjV:
                        newbestObjV = local_min
                        newbestChrom = populations[m][np.argmin(fitness_values[m])]

                # Check for improvement
                if newbestObjV < bestObjV:
                    bestObjV = newbestObjV
                    bestChrom = newbestChrom
                    gen0 = 0
                else:
                    gen0 += 1

                gen += 1

            # Save the best result for this subinterval
            results.append((sub_start, sub_end, bestObjV, bestChrom))

        # Transform results into a dictionary
        return {
            "subintervals": [
            {
                "sub_start": sub_start,
                "sub_end": sub_end,
                "bestObjV": bestObjV,
                "bestChrom": bestChrom.tolist()
            }
            for (sub_start, sub_end, bestObjV, bestChrom) in results
            ]
        }

    def generate_subintervals(self) -> None:
        """
        Generate subintervals based on the logic from the pseudo-code and the chosen frequency.

        Steps:
          - time_start = sample[0, 0]
          - time_end   = sample[-1, 0]
          - Derive parameters (three_weeks, six_weeks, one_week) from the frequency.
          - Compute delta = max((time_end - time_start)*0.75 / three_weeks, three_weeks)
          - sub_end from time_end down to time_end - six_weeks by one_week
          - sub_start from time_start to time_end - (time_end - time_start)/4 by delta
          - For each (sub_start, sub_end), filter the sample data.
        
        The resulting subintervals are stored in self.subintervals as a list of tuples:
          (sub_start, sub_end, sub_sample).
        """
        time_start = self.sample[0, 0]
        time_end = self.sample[-1, 0]

        if self.frequency == "daily":
            freq_list = [15, 30, 5]
        elif self.frequency == "weekly":
            freq_list = [3.0, 6.0, 1.0]
        elif self.frequency == "monthly":
            freq_list = [0.75, 1.5, 0.25]

        three_weeks, six_weeks, one_week = freq_list
        total_days = (time_end - time_start)
        delta = max((total_days * 0.75) / three_weeks, three_weeks)

        self.subintervals = []
        # Generate subintervals
        for sub_end in np.arange(time_end, time_end - six_weeks, -one_week):
            for sub_st in np.arange(time_start, time_end - total_days / 4, delta):
                mask = (self.sample[:, 0] >= sub_st) & (self.sample[:, 0] <= sub_end)
                sub_sample = self.sample[mask]
                if len(sub_sample) > 0:
                    self.subintervals.append((sub_st, sub_end, sub_sample))

    def initialize_population(self, param_bounds: np.ndarray, population_size: int) -> np.ndarray:
        """
        Wrapper for the Numba-compiled function to initialize a population.

        Parameters
        ----------
        param_bounds : np.ndarray, shape (D, 2)
            Bounds for the parameters [low, high].
        population_size : int
            Number of individuals in the population.

        Returns
        -------
        np.ndarray, shape (population_size, D)
            A randomly initialized population.
        """
        return njit_initialize_population(param_bounds, population_size)

    def selection(self, population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        """
        Wrapper for the Numba-compiled selection operation.

        Parameters
        ----------
        population : np.ndarray, shape (N, D)
            Population of N individuals, each with D parameters.
        fitness : np.ndarray, shape (N,)
            Fitness values for each individual.

        Returns
        -------
        np.ndarray, shape (N, D)
            The selected population of the same size N.
        """
        return njit_selection(population, fitness)

    def crossover(self, parents: np.ndarray, prob: float) -> np.ndarray:
        """
        Wrapper for the Numba-compiled single-point crossover.

        Parameters
        ----------
        parents : np.ndarray, shape (N, D)
            The current population of parents, each row is an individual.
        prob : float
            Probability to perform crossover for each pair.

        Returns
        -------
        np.ndarray, shape (N, D)
            The offspring population after crossover.
        """
        return njit_crossover(parents, prob)

    def mutate(self, offspring: np.ndarray, prob: float, param_bounds: np.ndarray) -> np.ndarray:
        """
        Wrapper for the Numba-compiled mutation operation.

        Parameters
        ----------
        offspring : np.ndarray, shape (N, D)
            Population to be mutated.
        prob : float
            Mutation probability in [0, 1].
        param_bounds : np.ndarray, shape (D, 2)
            Bounds for each parameter, used to resample mutated genes.

        Returns
        -------
        np.ndarray, shape (N, D)
            The mutated population (in-place, but also returned for convenience).
        """
        return njit_mutate(offspring, prob, param_bounds)

    def immigration_operation(self,
                              populations: list[np.ndarray],
                              fitness_values: list[np.ndarray]
                             ) -> list[np.ndarray]:
        """
        Wrapper for the Numba-compiled immigration operation.
        The best individual from population m replaces the worst individual in population m+1.

        Parameters
        ----------
        populations : list of np.ndarray
            A list of population arrays, e.g. populations[m].shape = (N, D).
        fitness_values : list of np.ndarray
            A list of fitness arrays, e.g. fitness_values[m].shape = (N,).

        Returns
        -------
        list of np.ndarray
            The updated list of populations after immigration.
        """
        return njit_immigration_operation(populations, fitness_values)

    def RSS(self, chromosome: np.ndarray, data: np.ndarray) -> float:
        """
        Wrapper for the Numba-compiled RSS function.

        Parameters
        ----------
        chromosome : np.ndarray, shape (4,)
            The nonlinear parameters [t_c, alpha, omega, phi].
        data : np.ndarray, shape (J, 2)
            The subinterval data, with column 0 = time, column 1 = price.

        Returns
        -------
        float
            The residual sum of squares for the fitted LPPL model on this data.
        """
        return njit_RSS(chromosome, data)

    def calculate_fitness(self, population: np.ndarray, data: np.ndarray) -> np.ndarray:
        """
        Wrapper for the Numba-compiled function that calculates fitness (RSS) 
        for each individual in the population.

        Parameters
        ----------
        population : np.ndarray, shape (N, 4)
            Each row is a chromosome [t_c, alpha, omega, phi].
        data : np.ndarray, shape (J, 2)
            The subinterval data, with time in data[:,0] and price in data[:,1].

        Returns
        -------
        np.ndarray, shape (N,)
            The RSS of each individual in the population.
        """
        return njit_calculate_fitness(population, data)