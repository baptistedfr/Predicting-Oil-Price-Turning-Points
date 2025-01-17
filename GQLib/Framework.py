import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from tqdm import tqdm
from datetime import datetime
import plotly.io as pio
from .Optimizers import MPGA, PSO, SGA, SA
from GQLib.Optimizers import Optimizer
from GQLib.LombAnalysis import LombAnalysis
from GQLib.Models import LPPL, LPPLS
from .enums import InputType

class Framework:
    """
    Framework for processing and analyzing financial time series using LPPL and Lomb-Scargle techniques.

    This framework includes:
    - Data loading and subinterval generation.
    - Optimization of LPPL parameters using a custom optimizer.
    - Lomb-Scargle periodogram analysis for detecting significant frequencies.
    - Visualization of results, including LPPL predictions and significant critical times.
    """

    def __init__(self, frequency: str = "daily", lppl_model: 'LPPL | LPPLS' = LPPL, input_type : InputType = InputType.WTI) -> None:
        """
        Initialize the Framework with a specified frequency for analysis.

        Parameters
        ----------
        frequency : str, optional
            The frequency of the time series data. Must be one of {"daily", "weekly", "monthly"}.
            Default is "daily".

        Raises
        ------
        ValueError
            If an invalid frequency is provided.
        """
        # Frequency validation and data loading
        if frequency not in ["daily", "weekly", "monthly"]:
            raise ValueError("The frequency must be one of 'daily', 'weekly', 'monthly'.")
        self.frequency = frequency
        self.input_type = input_type
        self.data = self.load_data(input_type)

        self.global_times = self.data[:, 0].astype(float)
        self.global_dates = self.data[:, 1]
        self.global_prices = self.data[:, 2].astype(float)

    def load_data(self) -> np.ndarray:
        """
        Load financial time series data from a CSV file.

        The CSV file is expected to have two columns:
        - "Date": Date of observation in the format "%m/%d/%Y".
        - "Price": Observed price.

        The function adds a numeric time index and returns a NumPy array.

        Returns
        -------
        np.ndarray
            A 2D array of shape (N, 3), where:
            - Column 0: Numeric time index (float).
            - Column 1: Dates as np.datetime64[D].
            - Column 2: Prices as float.
        """
        match self.input_type:
            case InputType.USO:
                data = pd.read_csv(f'data/USO_{self.frequency}.csv', sep=";")
                data['Price'] = data['Price'].apply(lambda x:x/8) # Stock split 1:8 en 2020
                data["Date"] = pd.to_datetime(data["Date"], format="%d/%m/%Y").values.astype("datetime64[D]")

            case InputType.WTI:
                data = pd.read_csv(f'data/WTI_Spot_Price_{self.frequency}.csv', skiprows=4)
                data.columns = ["Date", "Price"]
                data["Date"] = pd.to_datetime(data["Date"], format="%m/%d/%Y").values.astype("datetime64[D]")

            case InputType.SP500:
                data = pd.read_csv(f'data/sp500_Price_daily.csv', sep=";")
                data.columns = ["Date", "Price"]
                data["Date"] = pd.to_datetime(data["Date"], format="%m/%d/%Y").values.astype("datetime64[D]")
            
            case InputType.BTC : 
                data = pd.read_csv(f'data/BTC_{self.frequency}.csv', sep=",")
                data.columns = ["Date", "Price"]
                data["Date"] = pd.to_datetime(data["Date"], format="%Y-%m-%d").values.astype("datetime64[D]")

            case InputType.SSE:
                data = pd.read_csv(f'data/SSE_Price_{self.frequency}.csv', sep=";")
                data.columns = ["Date", "Price"]
                data["Date"] = pd.to_datetime(data["Date"], format="%m/%d/%Y").values.astype("datetime64[D]")

            case InputType.EURUSD : 
                data = pd.read_csv(f'data/EURUSD_{self.frequency}.csv', sep=";")
                data.columns = ["Date", "Price"]
                data["Date"] = pd.to_datetime(data["Date"], format="%Y-%m-%d").values.astype("datetime64[D]")
            
        # Date conversion and sorting
        data = data.sort_values(by="Date")
        # Add numeric time index
        t = np.linspace(0, len(data) - 1, len(data))
        data = np.insert(data.to_numpy(), 0, t, axis=1)
        return data


    def process(self, time_start: str, time_end: str, optimizer: Optimizer) -> dict:
        """
        Optimize LPPL parameters over multiple subintervals of the selected sample.

        Parameters
        ----------
        time_start : str
            Start date of the main sample in "%d/%m/%Y" format.
        time_end : str
            End date of the main sample in "%d/%m/%Y" format.
        optimizer : Optimizer
            Optimizer instance for parameter fitting
        Returns
        -------
        dict
            Optimization results for each subinterval.
        """
        # Configure the params of the optimizer based on the frequency
        optimizer.configure_params_from_frequency(self.frequency, optimizer.__class__.__name__)
        # Select data sample
        sample = self.select_sample(self.data, time_start, time_end) 

        # Generate subintervals
        subintervals = self.generate_subintervals(self.frequency, sample)

        # Store optimization results
        results = []

        # Optimize parameters for each subinterval
        for (sub_start, sub_end, sub_data) in tqdm(subintervals, desc="Processing subintervals", unit="subinterval"):
            
            bestObjV, bestParams = optimizer.fit(sub_start, sub_end, sub_data)
            results.append({
                "sub_start": sub_start,
                "sub_end": sub_end,
                "bestObjV": bestObjV,
                "bestParams": bestParams.tolist()
            })
        return results

    def analyze(self,
                results : dict = None,
                result_json_name: str = None,
                lppl_model: 'LPPL | LPPLS' = LPPL,
                significativity_tc : float = 0.3,
                use_package: bool = False,
                remove_mpf: bool = True,
                mpf_threshold: float = 1e-3,
                show: bool = False) -> dict:
        """
        Analyze results using Lomb-Scargle periodogram and identify significant critical times.

        Parameters
        ----------
        results : dict
            Optimization results to analyze.
        result_json_name : dict, optional
            Path to a JSON file containing results. If None, uses `self.results`.
        lppl_model : 'LPPL | LPPLS'
            Log Periodic Power Law Model utilized to computer the Lomb Periodogram
        significativity_tc : float
            Significance Threshold for Frequency Closeness. Default is 0.3
        use_package : bool
            Whether to use the astropy package to compute the Lomb Periodogram Power 
        remove_mpf : bool, optional
            Whether to remove the "most probable frequency" from the results. Default is True.
        mpf_threshold : float, optional
            Threshold for filtering frequencies close to the most probable frequency. Default is 1e-3.
        show : bool, optional
            Whether to display visualizations of the Lomb spectrum and LPPL fits. Default is False.
        Returns
        -------
        dict
            An updated list of results with significance flags.
        """
        if result_json_name is None and results is None:
            raise ValueError("Results must be provided.")

        if result_json_name is not None:
            with open(result_json_name, "r") as f:
                results = json.load(f)

        best_results = []

        # Visualizations if requested
        if show:
            num_intervals = len(results)
            num_cols = 3
            num_rows = (num_intervals + num_cols - 1) // num_cols
            fig, axes = plt.subplots(num_intervals, num_cols, figsize=(12, 6 * num_rows))

        for idx, res in enumerate(results):
            mask = (self.global_times >= res["sub_start"]) & (self.global_times <= res["sub_end"])
            t_sub = self.global_times[mask]
            y_sub = self.global_prices[mask]

            # Lomb-Scargle analysis
            lomb = LombAnalysis(lppl_model(t_sub, y_sub, res["bestParams"]))
            lomb.compute_lomb_periodogram(use_package=use_package)
            lomb.filter_results(remove_mpf=remove_mpf, mpf_threshold=mpf_threshold)
            is_significant = lomb.check_significance(significativity_tc=significativity_tc)

            if show:

                ax_residuals = axes[idx, 0]
                lomb.show_residuals(ax=ax_residuals)
                ax_residuals.set_title(f'Subinterval {idx + 1} Residuals')
            
                ax_spectrum = axes[idx, 1]
                lomb.show_spectrum(ax=ax_spectrum, use_filtered=True, show_threshold=True, highlight_freq=True)
                ax_spectrum.set_title(f'Subinterval {idx + 1} Spectrum (Significant: {is_significant})')

                ax_lppl = axes[idx, 2]
                self.show_lppl(lomb.lppl, ax=ax_lppl)
                ax_lppl.set_title(f'Subinterval {idx + 1} LPPL')


            # Add of the results
            best_results.append({
                "sub_start": res["sub_start"],
                "sub_end": res["sub_end"],
                "bestObjV": res["bestObjV"],
                "bestParams": res["bestParams"],
                "is_significant": is_significant,
                "power_value": max(lomb.power)
            })

        if show:
            plt.tight_layout()
            plt.show()

        return best_results

    def show_lppl(self, lppl: 'LPPL | LPPLS', ax=None, show: bool = False) -> None:
        """
        Visualize the LPPL or LPPLS fit alongside observed data.

        Parameters
        ----------
        lppl : LPPL or LPPLS
            An instance of the LPPL or LPPLS model with fitted parameters.
        ax : matplotlib.axes.Axes, optional
            An axis to plot on. If None, creates a new figure.
        show : bool, optional
            Whether to display the plot immediately. Default is False.
        """

        length_extended = (round(lppl.tc) + 1000) if self.frequency == "daily" else (round(lppl.tc) + 100) 

        # Calculate the maximum available length
        max_length = len(self.global_prices)

        # Adjust length_extended so it does not exceed the available length
        length_extended = min(length_extended, max_length)

        extended_t = np.arange(lppl.t[0], length_extended)
        extended_y = self.global_prices[int(extended_t[0]):int(extended_t[-1] + 1)]
        extended_dates = self.global_dates[int(extended_t[0]):int(extended_t[-1] + 1)]
        end_date = self.global_dates[int(lppl.t[-1])]

        lppl.t = extended_t
        predicted = lppl.predict(True)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(extended_dates, extended_y, label='Observed')
        ax.plot(extended_dates, predicted, label='Predicted')
        ax.axvline(x=end_date, color='r', linestyle='--', label='End of Subinterval')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.set_title('LPPL Model Prediction')
        ax.legend()

        if show:
            plt.show()

