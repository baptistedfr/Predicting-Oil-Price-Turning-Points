import numpy as np
from numba import njit
from matplotlib import pyplot as plt
from .njitFunc import njit_RSS

class LPPL:
    """
    A class to represent the Log-Periodic Power Law (LPPL) model.

    This class provides methods to fit the LPPL model to data, compute predictions,
    calculate residuals, and evaluate the goodness of fit using RSS.

    Attributes
    ----------
    t : np.ndarray
        Array of time points.
    y : np.ndarray
        Array of observed values (prices).
    tc : float
        Critical time parameter (t_c).
    omega : float
        Frequency of log-periodic oscillations.
    phi : float
        Phase shift of log-periodic oscillations.
    alpha : float
        Power-law exponent.
    A, B, C : float
        Linear coefficients computed during model fitting.
    residuals : np.ndarray or None
        Residuals of the model fit, computed after fitting.
    """

    def __init__(self, t, y, tc, omega, phi, alpha):
        """
        Initialize the LPPL model with time series data and parameters.

        Parameters
        ----------
        t : np.ndarray
            Array of time points.
        y : np.ndarray
            Array of observed values (prices).
        tc : float
            Critical time parameter (t_c).
        omega : float
            Frequency of log-periodic oscillations.
        phi : float
            Phase shift of log-periodic oscillations.
        alpha : float
            Power-law exponent.
        """
        self.t = t
        self.y = y
        self.tc = tc
        self.omega = omega
        self.phi = phi
        self.alpha = alpha

        self._compute_linear_params()
        self.residuals = None

    def __repr__(self):
        """
        Provide a detailed string representation of the LPPL model instance.
        """
        return f"LPPL(t={self.t}, y={self.y}, tc={self.tc}, omega={self.omega}, phi={self.phi}, alpha={self.alpha})"
    
    def __str__(self):
        """
        Provide a concise string representation of the LPPL model instance.
        """
        return f"LPPL(t={self.t}, y={self.y}, tc={self.tc}, omega={self.omega}, phi={self.phi}, alpha={self.alpha})"
    
    def show(self):
        """
        Plot the observed data and the fitted LPPL model.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(self.t, self.y, label="Data", color="black")
        plt.plot(self.t, self.predict(), label="Fit", color="red")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.legend()
        plt.show()

    def _compute_f_g(self) -> np.ndarray:
        """
        Compute the f(t) and g(t) components of the LPPL model.

        Returns
        -------
        f : np.ndarray
            Power-law component (t_c - t) ^ alpha.
        g : np.ndarray
            Log-periodic oscillation component (f * cos(...)).
        """
        dt = np.abs(self.tc - self.t)
        f = dt ** self.alpha
        g = f * np.cos(self.omega * np.log(dt) + self.phi)
        return f, g

    def _compute_linear_params(self) -> None:
        """
        Compute the linear parameters (A, B, C) of the LPPL model
        using least squares regression on the transformed variables.
        """
        f, g = self._compute_f_g()
        V = np.column_stack((np.ones_like(f), f, g))
        self.A, self.B, self.C = np.linalg.inv(V.T @ V) @ (V.T @ self.y)

    def predict(self, include_oscillation: bool = True) -> np.ndarray:
        """
        Predict values using the LPPL model.

        Parameters
        ----------
        include_oscillation : bool, optional
            If True, include the log-periodic oscillation term in the prediction.
            Default is True.

        Returns
        -------
        np.ndarray
            Predicted values based on the LPPL model.
        """
        f, g = self._compute_f_g()
        if include_oscillation:
            return self.A + self.B * f + self.C * g
        else:
            return self.A + self.B * f

    def compute_residuals(self, include_oscillation: bool = False) -> np.ndarray:
        """
        Compute the residuals of the LPPL model.

        Parameters
        ----------
        include_oscillation : bool, optional
            If True, include the log-periodic oscillation term in the residuals.
            Default is False.

        Returns
        -------
        np.ndarray
            Residuals (observed - predicted).
        """
        return self.y - self.predict(include_oscillation)

    def compute_rss(self) -> float:
        """
        Compute the Residual Sum of Squares (RSS) of the LPPL model.

        Returns
        -------
        float
            The RSS value.
        """
        self.compute_residuals(True)
        return np.sum(self.residuals ** 2)
    
    @staticmethod
    def numba_RSS(chromosome: np.ndarray, data: np.ndarray) -> float:
        """
        Compute the RSS for a given chromosome using a Numba-optimized function.

        Parameters
        ----------
        chromosome : np.ndarray
            Array representing the LPPL parameters [t_c, alpha, omega, phi].
        data : np.ndarray
            Observed data in the format [time, price].

        Returns
        -------
        float
            Residual Sum of Squares (RSS) value for the given parameters and data.
        """
        return njit_RSS(chromosome, data)