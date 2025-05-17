from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from GQLib.Models import LPPL, LPPLS
import logging
from statsmodels.tsa.stattools import adfuller

logging.getLogger(__name__)

class AbstractFilter(ABC):

    SEARCH_SPACE = {}

    @abstractmethod
    def filter(self, filter_params: Dict[str, Any]) -> bool:
        """
        Filter the time series estimation based on the model parameters.

        Parameters:
            filter_params (Dict[str, Any]): dictionary of params for the specific filter

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


    def filter(self, filter_params: Dict[str, Any]) -> bool:
        """
        Check if the model parameters are valid based on the filtering conditions.

        Parameters:
            filter_params (Dict[str, Any]): dictionary of filter parameters
                - linear_params: np.ndarray
                - non_linear_params: np.ndarray
                - t1: float
                - t2: float
                - prices: np.array

        Returns:
            bool: True if the parameters are valid, False otherwise.
        """
        linear_params = filter_params.get("linear_params", None)
        non_linear_params = filter_params.get("non_linear_params", None)
        t1 = filter_params.get("t1", None)
        t2 = filter_params.get("t2", None)
        prices = filter_params.get("prices", None)

        if any(param is None for param in [linear_params, non_linear_params, t1, t2, prices]):
            raise ValueError("Model parameters, time series, and bounds must be provided for filtering.")

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

        logging.debug("Model parameters: %s", model_params)

        logging.debug("Window size: %s", t2 - t1)
    
        return self._check_conditions(model_params, conditions)

    
    def _check_conditions(self, model_params: Dict[str, float], conditions: Dict[str, List[float]]) -> bool:
        """
        Check if the model parameters satisfy the filtering conditions.

        Parameters:
            model_params (Dict[str, float]): Model parameters to check.
            conditions (Dict[str, List[float]]): Filtering conditions.

        Returns:
            bool: True if the model parameters satisfy the conditions, False otherwise.
        """

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
        return (omega / 2 * np.pi) * np.log(np.abs((t_c - t1)/(t_c - t2)))

class LombFilter(AbstractFilter):
    
    SEARCH_SPACE = {
        "alpha": [0, 2.0],
        "omega": [1, 50],
        "t_c": [-0.2, 0.2],
        "phi": [0, 2 * np.pi],
        }
        
    def filter(self, filter_params: Dict[str, Any]) -> bool:
        """
        Filters the LPPL model's fit using the Lomb-Scargle periodogram analysis.

        This method computes the spectral power of the model's residuals and checks if the primary
        frequency peak is both statistically significant and sufficiently close to the target frequency,
        which is derived from the provided model parameters.

            filter_params (Dict[str, Any]): A dictionary containing the following keys:
                - non_linear_params (list): List of model parameters [t_c, omega, alpha].
                - t_series (np.ndarray): Time series data.
                - residuals (np.ndarray): Residuals of the LPPL model.
                - significance_level (float, optional): Significance level for the Lomb-Scargle test (default is 0.05).
                - significativity_tc (float, optional): Tolerance threshold for the target frequency (default is 0.1).

        Returns:
            bool: True if the highest power peak in the periodogram is close enough to the target frequency; otherwise, False.

        Raises:
            ValueError: If any of the required keys ('non_linear_params', 't_series', or 'residuals') are missing in filter_params.
        """

        self.params = filter_params.get("non_linear_params", None)
        if self.params is None:
            raise ValueError("Model parameters must be provided for Lomb-Scargle filtering.")
        self.t_series = filter_params.get("t_series", None)
        if self.t_series is None:
            raise ValueError("Time series must be provided for Lomb-Scargle filtering.")
        self.e = filter_params.get("residuals", None)
        if self.e is None:
            raise ValueError("Residuals must be provided for Lomb-Scargle filtering.")
        self.significance_level = filter_params.get("significance_level", 0.95)
        significativity_tc = filter_params.get("significativity_tc", 0.3)

        freqs, powers = self._compute_spectrum()
        target_freq = (self.params[2] if len(self.params) == 3 else self.params[3]) / (2 * np.pi)

        if len(self.params) == 4:
            logging.debug(f"Non-linear params -> t_c: {self.params[0]}, alpha: {self.params[1]}, phi: {self.params[2]}, omega: {self.params[3]}")
        elif len(self.params) == 3:
            logging.debug(f"Non-linear params -> t_c: {self.params[0]}, alpha: {self.params[1]}, omega: {self.params[2]}")
        logging.debug(f"Condition: {target_freq:.2f} +/- {significativity_tc:.2f}")
        logging.debug(f"Peak frequency: {freqs[np.argmax(powers)]:.2f}")
        logging.debug(f"Peak power: {powers.max():.2f}")

        if powers.size == 0:
            return False

        peak_idx = np.argmax(powers)
        peak_freq = freqs[peak_idx]

        lower = f"{(target_freq - significativity_tc):.2f}"
        val = f"{peak_freq:.2f}"
        upper = f"{(target_freq + significativity_tc):.2f}"
        lower_formatted = f"{lower:>8}"
        val_formatted = f"{val:>8}"
        upper_formatted = f"{upper:>8}"

        condition = abs(peak_freq - target_freq) < significativity_tc
        if condition:
            logging.debug(f"Condition for {'Frequency':<20}: {lower_formatted} <= {val_formatted} <= {upper_formatted} : True")
        else:
            logging.debug(f"Condition for {'Frequency':<20}: {lower_formatted} <= {val_formatted} <= {upper_formatted} : False")

        return condition

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
    
class StationarityFilter(AbstractFilter):
    """
    Filter based on the stationarity of the time series.
        -> The lppl model residuals should be stationary to be valid.
    """

    def filter(self, filter_params: Dict[str, Any]) -> bool:
        """
        Filter the time series estimation based on the model parameters.
        The time series is considered positively filtered if the ADF test p-value is less than 0.05.

        Parameters:
            filter_params (Dict[str, Any]): dictionary of filter parameters
                - model_params (list)
                - model_residuals (np.ndarray)

        Returns:
            bool: True if the time series is filtered, False otherwise
        """
        model_residuals = filter_params["model_residuals"]

        if adfuller(model_residuals)[1] < 0.05:
            return True
        else:
            return False
