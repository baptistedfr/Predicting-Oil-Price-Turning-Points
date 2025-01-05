import json
from typing import Tuple
import numpy as np
from scipy.optimize import minimize
from GQLib.Models import LPPL, LPPLS

from .abstract_optimizer import Optimizer


class NELDER_MEAD(Optimizer):

    def __init__(self, frequency: str, lppl_model: 'LPPL | LPPLS' = LPPLS) -> None:

        self.frequency = frequency
        self.lppl_model = lppl_model
        self.__name__ = self.__class__.__name__.replace("ABC", "")

        # Load optimization parameters from a JSON configuration file
        with open("params/params_mpga.json", "r") as f:
            params = json.load(f)

        self.PARAM_BOUNDS = params[f"{self.frequency.upper()}_PARAM_BOUNDS"]


    def fit(self, start: int, end: int, data: np.ndarray) -> Tuple[float, np.ndarray]:
        if self.lppl_model == LPPL:
            param_bounds = self.convert_param_bounds_lppl(end)
        elif self.lppl_model == LPPLS:
            param_bounds = self.convert_param_bounds_lppls(end)
        else:
            raise ValueError("Invalid model type.")

        def objective_function(params):
            return self.lppl_model.numba_RSS(params, data)

        def transform_params(params):
            # Transformation pour s'assurer que les paramètres restent dans les bornes
            transformed = np.empty_like(params)
            for i, (low, high) in enumerate(param_bounds):
                transformed[i] = low + (high - low) / (1 + np.exp(-params[i]))
            return transformed

        def inverse_transform_params(params):
            # Transformation inverse pour revenir à l'espace des paramètres d'origine
            inverse_transformed = np.empty_like(params)
            for i, (low, high) in enumerate(param_bounds):
                inverse_transformed[i] = np.log((params[i] - low) / (high - params[i]))
            return inverse_transformed

        initial_guess = np.mean(param_bounds, axis=1)
        transformed_initial_guess = inverse_transform_params(initial_guess)

        result = minimize(lambda x: objective_function(transform_params(x)), transformed_initial_guess, method='Nelder-Mead')

        bestObjV = result.fun
        bestParams = transform_params(result.x)

        return bestObjV, bestParams

    