import numpy as np
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from ..njitFunc import njit_RSS_LPPL

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

    def __init__(self, t, y, params):
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

        self.tc, self.omega, self.phi, self.alpha = params

        self._compute_linear_params()
        self.residuals = None

        self.__name__ = "LPPL"

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
    

    def hq_analysis(self, H=1.0, q=0.9):
        """
        Compute the (H, q)-analysis derivative:
        
            D^H_q f(x) = [ f(x) - f(q * x) ] / ( (1 - q) * x )^H

        where f(x) = ln(price) and x = t - tc.

        Parameters
        ----------
        H : float, optional
            The exponent in the (H, q)-analysis. Default is 1.0.
        q : float, optional
            The scale parameter in (H, q)-analysis, must be in (0,1). Default is 0.9.

        Returns
        -------
        x_valid : np.ndarray
            The array of valid x = t - tc (strictly positive).
        hq_values : np.ndarray
            The array of (H, q)-derivatives corresponding to x_valid.
        """

        dt = np.abs(self.tc - self.t)
        f = dt ** self.alpha
        f_q = (q * dt) ** self.alpha
        g = f * np.cos(self.omega * np.log((q * dt)) + self.phi)
        g_q = f * np.cos(self.omega * np.log((q * dt)) + self.phi)

        # Build design matrix
        V = np.column_stack((np.ones_like(f), f, g))
        # Attempt to invert (V^T V)

        params = np.linalg.inv(V.T @ V) @ (V.T @ self.y)
        A, B, C = params[0], params[1], params[2]

        V_q = np.column_stack((np.ones_like(f_q), f_q, g_q))

        params_q = np.linalg.inv(V_q.T @ V_q) @ (V_q.T @ self.y)
        A_q, B_q, C_q = params_q[0], params_q[1], params_q[2]
        
        predicted = A + B*f + C*g
        predicted_q = A_q + B_q*f_q + C_q*g_q
        
        return (predicted - q * predicted) / ((1 - q) * dt)**H
    
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
        return njit_RSS_LPPL(chromosome, data)