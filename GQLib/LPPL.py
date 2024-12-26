import numpy as np
from numba import njit
from matplotlib import pyplot as plt
from Lib.njitFunc import njit_RSS

class LPPL:

    def __init__(self, t, y, tc, omega, phi, alpha):
        self.t = t
        self.y = y
        self.tc = tc
        self.omega = omega
        self.phi = phi
        self.alpha = alpha

        self._compute_linear_params()
        self.residuals = None

    def __repr__(self):
        return f"LPPL(t={self.t}, y={self.y}, tc={self.tc}, omega={self.omega}, phi={self.phi}, alpha={self.alpha})"
    
    def __str__(self):
        return f"LPPL(t={self.t}, y={self.y}, tc={self.tc}, omega={self.omega}, phi={self.phi}, alpha={self.alpha})"
    
    def show(self):

        # Plot the data
        plt.figure(figsize=(12, 6))
        plt.plot(self.t, self.y, label="Data", color="black")
        plt.plot(self.t, self.predict(), label="Fit", color="red")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.legend()
        plt.show()

    def _compute_f_g(self) -> np.ndarray:
        dt = np.abs(self.tc - self.t)
        f = dt ** self.alpha
        g = f * np.cos(self.omega * np.log(dt) + self.phi)
        return f, g

    def _compute_linear_params(self) -> None:

        f, g = self._compute_f_g()

        V = np.column_stack((np.ones_like(f), f, g))

        self.A, self.B, self.C = np.linalg.inv(V.T @ V) @ (V.T @ self.y)

    def predict(self, include_oscillation: bool = True) -> np.ndarray:
        
        f, g = self._compute_f_g()
        
        if include_oscillation:
            return self.A + self.B * f + self.C * g
        else:
            return self.A + self.B * f

    def compute_residuals(self, include_oscillation: bool = False) -> None:
        return self.y - self.predict(include_oscillation)

    def compute_rss(self) -> float:
        self.compute_residuals(True)
        return np.sum(self.residuals ** 2)
    
    @staticmethod
    def numba_RSS(chromosome: np.ndarray, data: np.ndarray) -> float:
        return njit_RSS(chromosome, data)



