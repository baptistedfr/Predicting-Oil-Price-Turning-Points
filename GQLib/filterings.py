from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

class AbstractFilter(ABC):

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

class enculefilter(AbstractFilter):

    def filter():
        return True
    
class LPPLSConfidence(AbstractFilter):

    SEARCH_SPACE = {
        "alpha": [0, 2.0],
        "omega": [1, 50],
        "t_c": [],
    }

    CONDITIONS_EARLY_BUBBLE = {
        "alpha": [0.01, 1.2],
        "omega": [2, 25],
        "t_c": [-0.2, 0.2],
        "nb_oscillations": [2.5, np.inf],
        "damping": [0.0, np.inf],
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

    def __init__(self, model: 'LPPLS', len_window: int, **kwargs: Any) -> None:
        self.model = model
        self.len_window = len_window
        self.kwargs = kwargs

    def get_search_space(self) -> Dict[str, List[float]]:
        bounds = self.SEARCH_SPACE.copy()
        bounds["t_c"] = [self.len_window * b for b in bounds["t_c"]]
        if self.model is LPPL:
            bounds["phi"] = [0, 2 * np.pi]
        return self.SEARCH_SPACE
    
    def compute_damping(self, alpha: float, B: float, omega: float, C) -> float:
        """
        Compute the damping factor based on the LPPLS parameters.
        """
        return (alpha * omega) / (2 * np.pi) * np.exp(-omega * t_c)


lend = 100
len_window = 10
freq = 1
for i in range(0, lend - len_window, freq):
    print(i)