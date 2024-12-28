import numpy as np
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
from .LPPL import LPPL
import json

class LombAnalysis:
    """
    A class to perform Lomb-Scargle periodogram analysis on a time series.

    This class provides methods to compute, filter, and visualize the Lomb-Scargle
    periodogram for residuals of an LPPL model.
    """

    def __init__(self,
                 lppl: LPPL,
                 freqs: np.ndarray = np.linspace(0.0001, 20, 1000),
                 significance_level: float = 0.95):
        """
        Initialize the LombAnalysis instance.

        Parameters
        ----------
        lppl : LPPL
            An LPPL instance containing the time series data and residuals.
        freqs : np.ndarray
            Array of candidate frequencies for Lomb-Scargle analysis.
        significance_level : float
            Statistical significance level (default is 0.95).

        Raises
        ------
        ValueError
            If the frequency array is empty.
        """
        if len(freqs) == 0:
            raise ValueError("The frequency array cannot be empty.")

        self.lppl = lppl
        self.x = self.lppl.compute_residuals(False)
        self.freqs = freqs
        self.significance_level = significance_level

        self.target_freq = lppl.omega / (2 * np.pi)

        self.power = None  # Will hold the Lomb-Scargle power array
        self.critical_value = None  # Will store the significance threshold
        self.filtered_freqs = None  # After filtering
        self.filtered_power = None

    def compute_ln_tc_tau(self) -> np.ndarray:

        dt = np.abs(self.lppl.tc - self.lppl.t)
        return np.log(dt)


    def show_residuals(self, ax=None, show: bool = False) -> None:
        """
        Visualize the residuals of the LPPL model.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Matplotlib axis object to use for the plot (default: None).
        show : bool
            Whether to display the plot (default: False).
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        test = self.lppl.compute_residuals(False)

        # test to list into a json
        test_list = test.tolist()

        with open('residuals.json', 'w') as f:
            json.dump(test_list, f)

        ax.plot(self.new_t, self.lppl.compute_residuals(False), label="Residuals without Oscillation", color="blue")
        ax.plot(self.new_t, self.lppl.compute_residuals(True), label="Residuals with Oscillation", color="red")
        ax.set_xlabel("Time (ln(tc - t))")
        ax.set_ylabel("Residuals")
        ax.set_title("LPPL Residuals")
        ax.legend()

        if show:
            plt.show()
    
    def compute_lomb_periodogram(self, use_package: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the Lomb-Scargle power spectrum for the residual time series.

        Parameters
        ----------
        use_package : bool
            If True, use the Astropy LombScargle implementation. Otherwise, use a custom implementation.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Frequencies and their corresponding power values.

        Raises
        ------
        ValueError
            If the time series or residuals are not properly initialized.
        """

        self.new_t = self.compute_ln_tc_tau()

        if use_package:
            ls = LombScargle(self.new_t, self.x)
            self.power = ls.power(self.freqs)
        else:
            J = len(self.new_t)
            mean_x = 1/J * np.sum(self.x)  # Mean of x
            var_x = 1/(J-1) * np.sum((self.x - mean_x) ** 2)  # Variance of x
            self.power = []

            for f in self.freqs:
                sin_term = np.sum(np.sin(4 * np.pi * f * self.new_t))
                cos_term = np.sum(np.cos(4 * np.pi * f * self.new_t))
                t_offset = (1 / (4 * np.pi * f)) * np.arctan(sin_term / cos_term)

                cos_num = np.sum((self.x - mean_x) * np.cos(2 * np.pi * f * (self.new_t - t_offset))) ** 2
                cos_denom = np.sum(np.cos(2 * np.pi * f * (self.new_t - t_offset)) ** 2)
                sin_num = np.sum((self.x - mean_x) * np.sin(2 * np.pi * f * (self.new_t - t_offset))) ** 2
                sin_denom = np.sum(np.sin(2 * np.pi * f * (self.new_t - t_offset)) ** 2)
                power_value = (cos_num / cos_denom + sin_num / sin_denom) / (2 * var_x)
                self.power.append(power_value)

            self.power = np.array(self.power)

        M = len(self.freqs)
        self.critical_value = -np.log(1.0 - (1.0 - self.significance_level) ** (1.0 / M))

        return self.freqs, self.power

    def filter_results(self,
                       remove_mpf: bool = True,
                       mpf_threshold: float = 1e-3,
                       mpf_factor: float = 1.5
                       ) -> tuple[np.ndarray, np.ndarray]:
        """
        Filter the frequencies and their power based on validity criteria.

        Parameters
        ----------
        remove_mpf : bool
            Whether to remove the most probable frequency (default: True).
        mpf_threshold : float
            Threshold for filtering frequencies near the most probable frequency (default: 1e-3).
        mpf_factor : float
            Factor used to calculate the most probable frequency (default: 1.5).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Filtered frequencies and their corresponding power values.

        Raises
        ------
        RuntimeError
            If the Lomb-Scargle power spectrum is not computed.
        """
        if self.power is None:
            raise RuntimeError("You must run compute_lomb_periodogram() before filtering results.")

        N = len(self.x)
        mpf = mpf_factor / N

        f_out = []
        p_out = []

        for (f, pwr) in zip(self.freqs, self.power):
            if remove_mpf and abs(f - mpf) < mpf_threshold:
                continue
            if pwr < self.critical_value:
                continue
            f_out.append(f)
            p_out.append(pwr)

        self.filtered_freqs = np.array(f_out)
        self.filtered_power = np.array(p_out)

        return self.filtered_freqs, self.filtered_power
    
    def check_significance(self) -> bool:
        """
        Check if the target frequency is statistically significant.

        Returns
        -------
        bool
            True if the target frequency is significant, False otherwise.

        Raises
        ------
        RuntimeError
            If the power spectrum or filtered results are not available.
        """
        if self.power is None:
            raise RuntimeError("No Lomb-Scargle power computed. Call compute_lomb_periodogram() first.")

        if self.critical_value is None:
            raise RuntimeError("No significance threshold computed. Call compute_lomb_periodogram() first.")

        if self.filtered_freqs is None or self.filtered_power is None:
            raise RuntimeError("No filtered results found. Call filter_results() first.")

        idx = np.argmax(self.filtered_power)
        return abs(self.filtered_freqs[idx] - self.target_freq) < 0.3

    def show_spectrum(self,
                      ax=None,
                      show: bool = False,
                      use_filtered: bool = False,
                      show_threshold: bool = False,
                      show_max_power: bool = False,
                      highlight_freq: bool = False) -> None:
        """
        Visualize the Lomb-Scargle power spectrum.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Matplotlib axis object to use for the plot (default: None).
        show : bool
            Whether to display the plot (default: False).
        use_filtered : bool
            Whether to plot only the filtered frequencies and power (default: False).
        show_threshold : bool
            Whether to show the significance threshold (default: False).
        show_max_power : bool
            Whether to highlight the frequency with the maximum power (default: False).
        highlight_freq : bool
            Whether to highlight the target frequency (default: False).

        Raises
        ------
        RuntimeError
            If the power spectrum is not computed.
        """
        if self.power is None:
            raise RuntimeError("No Lomb-Scargle power computed. Call compute_lomb_periodogram() first.")

        if use_filtered and (self.filtered_freqs is None or self.filtered_power is None):
            raise RuntimeError("No filtered results found. Call filter_results() first.")

        if use_filtered:
            freqs_plot = self.filtered_freqs
            power_plot = self.filtered_power
            title = "Lomb-Scargle (Filtered)"
        else:
            freqs_plot = self.freqs
            power_plot = self.power
            title = "Lomb-Scargle (All Frequencies)"

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(freqs_plot, power_plot, label="Lomb-Scargle Power", color="blue")
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Power")
        ax.set_title(title)
  

        if show_threshold and self.critical_value is not None:
            ax.axhline(y=self.critical_value, color="red", linestyle="--",
                        label=f"Significance threshold={self.critical_value:.2f}")

        if highlight_freq:
            ax.axvline(x=self.target_freq, color="green", linestyle=":", 
                        label=f"Highlighted freq={self.target_freq:.4f}")
            ax.axvline(x=self.target_freq + 0.05, color="gray", linestyle=":")
            ax.axvline(x=self.target_freq - 0.05, color="gray", linestyle=":")

        if show_max_power:
            idx_peak = np.argmax(power_plot)
            best_freq = freqs_plot[idx_peak]
            ax.axvline(x=best_freq, color="orange", linestyle="--", label="Max Power")
            

        ax.legend()

        if show:
            plt.show()

    def run(self, use_filtered: bool = True, use_package: bool = False) -> None:
        """
        Compute, filter, and visualize the Lomb-Scargle power spectrum.

        Parameters
        ----------
        use_filtered : bool
            Whether to plot only the filtered results (default: True).
        use_package : bool
            Whether to use the Astropy library for Lomb-Scargle computation (default: False).
        """
        self.compute_lomb_periodogram(use_package=use_package)
        self.filter_results()
        self.show_spectrum(use_filtered=use_filtered)