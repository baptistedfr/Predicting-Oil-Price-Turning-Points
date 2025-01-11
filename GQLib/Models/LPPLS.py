import numpy as np
from matplotlib import pyplot as plt
from ..njitFunc import njit_RSS_LPPLS

class LPPLS:
    """
    A class to represent the Log-Periodic Power Law (LPPLS) model with 4 linear parameters.

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
    alpha : float
        Power-law exponent.
    A, B, C1, C2 : float
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
        alpha : float
            Power-law exponent.
        """
        self.t = t
        self.y = y #np.log(y)  # Log-transform the observed prices

        self.tc, self.omega, self.alpha = params

        self._compute_linear_params()
        self.residuals = None

        self.__name__ = "LPPLS"

    def __repr__(self):
        return f"LPPL(t={self.t}, y={self.y}, tc={self.tc}, omega={self.omega}, alpha={self.alpha})"

    def show(self):
        """
        Plot the observed data and the fitted LPPL model.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(self.t, self.y, label="Data", color="black")
        plt.plot(self.t, self.predict(), label="Fit", color="red")
        plt.xlabel("Time")
        plt.ylabel("Log(Price)")
        plt.legend()
        plt.show()

    def _compute_f_g_h(self):
        """
        Compute the f(t), g(t), and h(t) components of the LPPL model.

        Returns
        -------
        f : np.ndarray
            Power-law component (t_c - t)^alpha.
        g : np.ndarray
            Cosine component (f * cos(...)).
        h : np.ndarray
            Sine component (f * sin(...)).
        """
        dt = np.abs(self.tc - self.t)
        f = dt ** self.alpha
        g = f * np.cos(self.omega * np.log(dt))
        h = f * np.sin(self.omega * np.log(dt))
        return f, g, h

    def _compute_linear_params(self):
        """
        Compute the linear parameters (A, B, C1, C2) of the LPPL model
        using least squares regression on the transformed variables.
        """
        f, g, h = self._compute_f_g_h()
        V = np.column_stack((np.ones_like(f), f, g, h))
        try:
            params = np.linalg.inv(V.T @ V) @ (V.T @ self.y)
            self.A, self.B, self.C1, self.C2 = params
        except np.linalg.LinAlgError:
            self.A, self.B, self.C1, self.C2 = np.nan, np.nan, np.nan, np.nan

    def predict(self, include_oscillation: bool = True) -> np.ndarray:
        """
        Predict values using the LPPL model.

        Parameters
        ----------
        include_oscillation : bool, optional
            If True, include the log-periodic oscillation terms in the prediction.
            Default is True.

        Returns
        -------
        np.ndarray
            Predicted values based on the LPPL model.
        """
        f, g, h = self._compute_f_g_h()
        if include_oscillation:
            return self.A + self.B * f + self.C1 * g + self.C2 * h
        else:
            return self.A + self.B * f

    def compute_residuals(self, include_oscillation: bool = True) -> np.ndarray:
        """
        Compute the residuals of the LPPL model.

        Parameters
        ----------
        include_oscillation : bool, optional
            If True, include the log-periodic oscillation terms in the residuals.
            Default is True.

        Returns
        -------
        np.ndarray
            Residuals (observed - predicted).
        """
        self.residuals = self.y - self.predict(include_oscillation)
        return self.residuals

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
            Array representing the LPPL parameters [t_c, alpha, omega, A, B, C1, C2].
        data : np.ndarray
            Observed data in the format [time, log(price)].

        Returns
        -------
        float
            Residual Sum of Squares (RSS) value for the given parameters and data.
        """
        return njit_RSS_LPPLS(chromosome, data)
