from typing import Tuple
import numpy as np
from GQLib.Models import LPPL, LPPLS
import json
from .abstract_optimizer import Optimizer
from ..njitFunc import (
    njit_update_position,
    njit_update_velocity
)


class FA(Optimizer):
    """
    Firefly Optimisation for optimizing LPPL parameters.

    The Firefly Algorithm is a metaheuristic optimization algorithm inspired by the social behavior of fireflies. 
    It uses the concept of attraction based on brightness and global communication among particles to solve optimization problems 
    effectively.
    It is based on the swarm behavior like fish, insects, and bird training in nature. 
    Firefly algorithm has a lot of similarities with SI algorithms like PSO, Artificial Bee Colony optimization, and Bacterial Foraging algorithms. 
    Firefly algorithm uses real random numbers. It is based on the global communication among the swarming particles (i.e., the fireflies). It appears to be more effective in multiobjective optimization. 
    The firefly algorithm has three rules which are based on flashing characteristics of real fireflies. These are as follows:
    1. All fireflies are unisex, and they will move toward more attractive and brighter ones regardless their sex.
    2. The degree of attraction of a firefly is proportional to its brightness which cuts as the distance from the other firefly 
    increases due to the fact that the air absorbs light. If there is not a brighter or more attractive firefly than a particular one, 
    it will then move randomly.
    3. The brightness or light intensity of a firefly is determined by the value of the objective function of a given problem.
    """

    def __init__(self, lppl_model: 'LPPL | LPPLS' = LPPL, beta0: float = 1.0, gamma: float = 0.8, alpha: float =0.2) -> None:
        """
        Initialize the FA optimizer.

        Parameters
        ----------
        lppl_model : LPPL | LPPLS
            Model used for RSS computation.
        beta0 : float
            Initial attractiveness (default: 1.0).
        gamma : float
            Light absorption coefficient (default: 0.8).
        alpha : float
            Randomization parameter (default: 0.2).
        NUM_FIREFLIES : int
            Number of fireflies (default: 10).
        MAX_GEN : int
            Maximum number of generations (default: 100).
        """
        self.lppl_model = lppl_model

        self.beta0 = beta0 # Attractiveness
        self.gamma = gamma # Light absorption coefficient
        self.alpha = alpha # Randomization parameter
        
        self.NUM_FIREFLIES = None
        self.MAX_GEN = None

    def fit(self, start: int, end: int, data: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Fit LPPL parameters

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
        else:
            raise ValueError("Invalid model type.")
        num_params = param_bounds.shape[0]
        self.fitness_history = []

        # Initialize fireflies with initial fitness values
        fireflies = [
            np.array([np.random.uniform(low, high) for low, high in param_bounds])
            for _ in range(self.NUM_FIREFLIES)
        ]
        # Initialise the fitness of the fireflies
        fitness = [self.lppl_model.numba_RSS(firefly, data) for firefly in fireflies]
        best_index = np.argmin(fitness)
        best_fitness = fitness[best_index]
        # Iterate through the generations
        for _ in range(self.MAX_GEN):
            for i in range(self.NUM_FIREFLIES):
                for j in range(self.NUM_FIREFLIES):
                    if fitness[j] < fitness[i]:
                        # Calculate the distance
                        r = np.linalg.norm(fireflies[i] - fireflies[j])
                        attractiveness = np.exp(-self.gamma * r**2)
                        fireflies[i] += self.beta0 * attractiveness * (fireflies[j] - fireflies[i]) + \
                            self.alpha * (np.random.rand(num_params) - 0.5)
                        # On limite les positions dans la borne
                        for k in range(len(fireflies[i])):
                            lower_bound, upper_bound = param_bounds[k]
                            fireflies[i][k] = max(lower_bound, min(fireflies[i][k], upper_bound))
                        fitness[i] = self.lppl_model.numba_RSS(fireflies[i], data)

                        # Update du meilleur firefly
                        if fitness[i] < best_fitness:
                            best_fitness = fitness[i]
                            best_index = i

            self.fitness_history.append(best_fitness)

        best_firefly = fireflies[best_index]
        return best_fitness, best_firefly
       
    

    