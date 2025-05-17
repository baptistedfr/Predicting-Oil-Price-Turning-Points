from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from GQLib.Models import LPPL, LPPLS
import logging

logging.getLogger(__name__)

class AbstractFilter(ABC):

    SEARCH_SPACE = {}

    @abstractmethod
    def filter(self, model_params: list) -> bool:
        """
        Filter the time series estimation based on the model parameters.

        Parameters:
            model_params (list): list of model parameters [t_c, omega, alpha]

        Returns:
            bool: True if the time series is filtered, False otherwise
        """
        pass
    
    def get_search_space(self, t1: int, t2: int) -> Dict[str, List[float]]:
        """
        Get the search space for the model parameters, i.e. the bounds for the parameters.

        Parameters:
            t1 (int): start time of the interval
            t2 (int): end time of the interval
        
        Returns:
            Dict[str, List[float]]: search space for the model parameters
        """
        bounds = self.SEARCH_SPACE.copy()
        bounds["t_c"] = self.adapt_bounds(bounds["t_c"], t1, t2)
        return bounds
    
    @staticmethod
    def adapt_bounds(t_c_bound: list[float], t1: float, t2: float) -> Tuple[float, float]:
        """
        Adapt the t_c bounds based on the time interval [t1, t2].

        Parameters:
            t_c_bound (list[float]): bounds for t_c
            t1 (float): start time of the interval
            t2 (float): end time of the interval

        Returns:
            Tuple[float, float]: adapted bounds for t_c
        """
        t_c_bound = [t_c_bound[0] * (t2 - t1) + t2, t_c_bound[1] * (t2 - t1) + t2]
        return t_c_bound
    
class LPPLSConfidence(AbstractFilter):

    SEARCH_SPACE = {
    "alpha": [0, 2.0],
    "omega": [1, 50],
    "t_c": [-0.2, 0.2],
    "phi": [0, 2 * np.pi],
    }

    CONDITIONS_EARLY_BUBBLE = {
        "alpha": [0.01, 1.2],
        "omega": [2, 25],
        "t_c": [-0.05, 0.1],
        "nb_oscillations": [2.5, np.inf],
        "damping": [0.0, np.inf],
    }

    CONDITIONS_BUBBLE_END = {
        "alpha": [0.01, 1.2],
        "omega": [2, 25],
        "t_c": [-0.05, 0.1],
        "nb_oscillations": [2.5, np.inf],
        "damping": [0.0, np.inf],
    }

    def __init__(self) -> None:
        super().__init__()


    def filter(self, linear_params: np.ndarray, non_linear_params: np.ndarray, t1, t2, prices) -> bool:
        """
        Check if the model parameters are valid based on the filtering conditions.

        Parameters:
            linear_params (np.ndarray): Linear parameters of the model.
            non_linear_params (np.ndarray): Non-linear parameters of the model.

        Returns:
            bool: True if the parameters are valid, False otherwise.
        """
        
        if len(linear_params) == 4:
            A, B, C1, C2 = linear_params
            C = self.compute_C(C1, C2)
        elif len(linear_params) == 3:
            A, B, C = linear_params

        if len(non_linear_params) == 4:
            t_c, alpha, omega, _ = non_linear_params
        elif len(non_linear_params) == 3:
            t_c, alpha, omega = non_linear_params

        conditions = self.CONDITIONS_EARLY_BUBBLE.copy()
        conditions["t_c"] = self.adapt_bounds(conditions["t_c"], t1, t2)

        conditions["t_c"] = [t - t2 for t in conditions["t_c"]]

        model_params = {
            "alpha": alpha,
            "omega": omega,
            "t_c": t_c -t2,
            "nb_oscillations": self.compute_nb_oscillations(t_c, t1, t2, omega),
            "damping": self.compute_dampling(alpha, B, omega, C),
        }

        sign = np.median(prices / prices[0] - 1)

        logging.debug("Model parameters: %s", model_params)

        logging.debug("Window size: %s", t2 - t1)
    
        if self._check_conditions(model_params, conditions):
            if sign > 0:
                return 1
            else:
                return -1
        else:
            return 0
    
    def _check_conditions(self, model_params: Dict[str, float], conditions: Dict[str, List[float]]) -> bool:
        """
        Check if the model parameters satisfy the filtering conditions.

        Parameters:
            model_params (Dict[str, float]): Model parameters to check.
            conditions (Dict[str, List[float]]): Filtering conditions.

        Returns:
            bool: True if the model parameters satisfy the conditions, False otherwise.
        """
        for param, bounds in conditions.items():
            all_valid = True
            for param, bounds in conditions.items():
                value = model_params[param]
                lower = f"{bounds[0]:.2f}"
                val = f"{value:.2f}"
                upper = f"{bounds[1]:.2f}"
                param_formatted = f"{param:<20}"  # left-align parameter name in a 20-char field
                lower_formatted = f"{lower:>8}"   # right-align bounds and value in an 8-char field
                val_formatted = f"{val:>8}"
                upper_formatted = f"{upper:>8}"
                if bounds[0] <= value <= bounds[1]:
                    logging.debug(f"Condition for {param_formatted}: {lower_formatted} <= {val_formatted} <= {upper_formatted} : True")
                else:
                    logging.debug(f"Condition for {param_formatted}: {lower_formatted} <= {val_formatted} <= {upper_formatted} : False")
                    all_valid = False
            return all_valid
    
    def compute_C(self, C1: float, C2: float) -> float:
        """
        Compute the C parameter based on C1 and C2.

        Parameters:
            C1 (float): First component of the model.
            C2 (float): Second component of the model.

        Returns:
            float: Computed C parameter.
        """
        return np.sqrt(C1**2 + C2**2)
    
    def compute_dampling(self, alpha: float, B: float, omega: float, C) -> float:
        """
        Compute the damping factor based on the LPPLS parameters.
        """
        return (alpha * np.abs(B)) / (omega * np.abs(C))
    
    def compute_nb_oscillations(self, t_c: float, t1: float, t2: float, omega: float) -> float:
        """
        Compute the number of oscillations based on the LPPLS parameters.
        """
        return (omega) * np.log(np.abs((t_c - t1)/(t2 - t1)))

class LombFilter(AbstractFilter):
    
    SEARCH_SPACE = {
        "alpha": [0, 2.0],
        "omega": [1, 50],
        "t_c": [-0.2, 0.2],
        "phi": [0, 2 * np.pi],
        }
        
    def filter(
        self,
        model_params: list,
        residuals: np.ndarray,
        t_series: np.ndarray,
        significance_level: float = 0.95,
        significativity_tc: float = 0.3
    ) -> bool:
        """
        Filter the fit of the LPPL model based on the Lomb-Scargle periodogram.
        Returns True if the main peak is significant and close to target frequency.

        model_params: [t_c, omega, alpha]
        significativity_tc: tolerance around target frequency
        """
        self.params = model_params
        self.t_series = t_series
        self.e = residuals
        self.significance_level = significance_level

        freqs, powers = self._compute_spectrum()
        target_freq = self.params[1] / (2 * np.pi)

        if powers.size == 0:
            return False

        peak_idx = np.argmax(powers)
        peak_freq = freqs[peak_idx]

        return abs(peak_freq - target_freq) < significativity_tc

    def _compute_spectrum(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Lomb-Scargle power spectrum for residuals.
        """
        tc = self.params[0]
        dt = np.abs(self.t_series - tc)
        e = self.e

        mean_e = np.mean(e)
        var_e = np.var(e, ddof=1)

        freqs = np.linspace(0.0001, 20, 1000)
        powers = np.array([self._compute_power(f, dt, mean_e, var_e) for f in freqs])

        return self._lomb_scargle_filter(freqs, powers)

    def _compute_power(self, f: float, dt: np.ndarray, mean_e: float, var_e: float) -> float:
        """
        Compute Lomb-Scargle power for one frequency.
        """
        omega = 2 * np.pi * f
        sin_sum = np.sum(np.sin(omega * dt))
        cos_sum = np.sum(np.cos(omega * dt))
        tau = np.arctan2(sin_sum, cos_sum) / omega

        e_centered = self.e - mean_e

        arg = omega * (dt - tau)
        cos_arg = np.cos(arg)
        sin_arg = np.sin(arg)

        cos_num = np.sum(e_centered * cos_arg) ** 2
        cos_den = np.sum(cos_arg ** 2)
        sin_num = np.sum(e_centered * sin_arg) ** 2
        sin_den = np.sum(sin_arg ** 2)

        return (cos_num / cos_den + sin_num / sin_den) / (2 * var_e)

    def _lomb_scargle_filter(
        self,
        frequencies: np.ndarray,
        powers: np.ndarray,
        remove_mpf: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter frequencies by false-alarm level and optional MPF removal.
        """
        M = len(frequencies)
        critical = -np.log(1 - (1 - self.significance_level) ** (1.0 / M))

        mask = powers >= critical
        if remove_mpf:
            mpf = 1.5 / len(self.e)
            mask &= np.abs(frequencies - mpf) > 1e-3

        return frequencies[mask], powers[mask]