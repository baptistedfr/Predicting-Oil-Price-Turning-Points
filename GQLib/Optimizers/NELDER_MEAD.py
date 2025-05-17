from typing import Tuple
import numpy as np
from scipy.optimize import minimize
from GQLib.Models import LPPL, LPPLS

from .abstract_optimizer import Optimizer
import logging

logging.getLogger(__name__)

class NELDER_MEAD(Optimizer):

    def __init__(self, lppl_model: 'LPPL | LPPLS' = LPPL) -> None:

        self.lppl_model = lppl_model
        
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

        result = minimize(lambda x: objective_function(transform_params(x)),
                  transformed_initial_guess, 
                  method='Nelder-Mead',
                  options={'maxiter': 10000, 'maxfev': 15000, 'fatol': 1e-5, 'xatol': 1e-5})

        bestObjV = result.fun
        bestParams = transform_params(result.x)

        return bestObjV, bestParams

    # def fit(self, start: int, end: int, data: np.ndarray) -> Tuple[float, np.ndarray]:
    #     """
    #     Fit LPPL or LPPLS model to the data using constrained optimization (L-BFGS-B).
        
    #     Parameters
    #     ----------
    #     start : int
    #         Start index of the fitting interval.
    #     end : int
    #         End index of the fitting interval.
    #     data : np.ndarray
    #         Log-price time series to fit.

    #     Returns
    #     -------
    #     Tuple[float, np.ndarray]
    #         The best objective value and the estimated parameters.
    #     """
    #     # Param bounds matrix (N, 2)
    #     if self.lppl_model == LPPL:
    #         param_bounds = self.convert_param_bounds_lppl(end)
    #         param_names = ["t_c", "omega", "phi", "alpha"]
    #     elif self.lppl_model == LPPLS:
    #         param_bounds = self.convert_param_bounds_lppls(end)
    #         param_names = ["t_c", "omega", "alpha"]
    #     else:
    #         raise ValueError("Invalid model type.")

    #     # Initial guess: mean of bounds
    #     initial_guess = np.mean(param_bounds, axis=1)

    #     # Build x0 as a dictionary (not strictly needed but mirrors external code structure)
    #     init_param_dict = {name: val for name, val in zip(param_names, initial_guess)}

    #     # Pack initial guess into list format
    #     x0 = [init_param_dict[param] for param in param_names]

    #     # Build bounds for scipy.optimize
    #     bounds_list = [tuple(b) for b in param_bounds]

    #     # Time vector
    #     t = np.arange(len(data))

    #     # Define the objective function wrapper
    #     def objective(params, *args):
    #         t, log_prices = args
    #         return self.lppl_model.numba_RSS(params, log_prices)

    #     # Run constrained optimization with L-BFGS-B
    #     result = minimize(
    #         fun=objective,
    #         x0=np.array(x0),
    #         args=(t, data),
    #         bounds=bounds_list,
    #         method='L-BFGS-B',
    #         options={
    #             'ftol': 1e-12,
    #             'gtol': 1e-12,
    #             'maxiter': 10000,
    #             'maxfun': 15000,
    #             'disp': False
    #         }
    #     )

    #     logging.warning("Optimization result: %s", result)

    #     return result.fun, result.x
            