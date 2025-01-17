import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

from GQLib.Optimizers import Optimizer
from GQLib.LombAnalysis import LombAnalysis
from GQLib.Models import LPPL, LPPLS

class Framework:
    """
    Framework for processing and analyzing financial time series using LPPL and Lomb-Scargle techniques.

    This framework includes:
    - Data loading and subinterval generation.
    - Optimization of LPPL parameters using a custom optimizer.
    - Lomb-Scargle periodogram analysis for detecting significant frequencies.
    - Visualization of results, including LPPL predictions and significant critical times.
    """

    def __init__(self, frequency: str = "daily", lppl_model: 'LPPL | LPPLS' = LPPL) -> None:
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

        self.lppl_model = lppl_model

        self.data = self.load_data()

        self.global_times = self.data[:, 0].astype(float)
        self.global_dates = self.data[:, 1]
        self.global_prices = self.data[:, 2].astype(float)

        self.stop = False

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
        data = pd.read_csv(f'./data/WTI_Spot_Price_{self.frequency}.csv', skiprows=4)
        data.columns = ["Date", "Price"]

        # Date conversion and sorting
        data["Date"] = pd.to_datetime(data["Date"], format="%m/%d/%Y").values.astype("datetime64[D]")
        data = data.sort_values(by="Date")

        # Add numeric time index
        t = np.linspace(0, len(data) - 1, len(data))
        data = np.insert(data.to_numpy(), 0, t, axis=1)

        return data

    def select_sample(self) -> np.ndarray:
        """
        Select a sample from the global time series based on a user-defined date range.

        Returns
        -------
        np.ndarray
            A 2D array of shape (M, 2), where:
            - Column 0: Numeric time indices (float).
            - Column 1: Prices (float).
        """
        # Convert start and end dates to datetime64
        start_dt = np.datetime64(pd.to_datetime(self.time_start, format="%d/%m/%Y"))
        end_dt = np.datetime64(pd.to_datetime(self.time_end, format="%d/%m/%Y"))

        # Filter rows within the specified date range
        mask = (self.data[:, 1] >= start_dt) & (self.data[:, 1] <= end_dt)
        sample = self.data[mask]

        return sample[:, [0, 2]].astype(float)

    def process(self, time_start: str, time_end: str, optimizer_class: Optimizer) -> None:
        """
        Optimize LPPL parameters over multiple subintervals of the selected sample.

        Parameters
        ----------
        time_start : str
            Start date of the main sample in "%d/%m/%Y" format.
        time_end : str
            End date of the main sample in "%d/%m/%Y" format.
        optimizer_class : Optimizer
            A class for optimizing LPPL parameters.
        """
        self.time_start = time_start
        self.time_end = time_end
        self.sample = self.select_sample()

        # Generate subintervals
        self.generate_subintervals()

        # Store optimization results
        self.results = []

        # Optimize parameters for each subinterval
        for (sub_start, sub_end, sub_data) in tqdm(self.subintervals, desc="Processing subintervals", unit="subinterval"):
            optimizer = optimizer_class(self.frequency, self.lppl_model)
            bestObjV, bestParams = optimizer.fit(sub_start, sub_end, sub_data)
            self.results.append({
                "sub_start": sub_start,
                "sub_end": sub_end,
                "bestObjV": bestObjV,
                "bestParams": bestParams.tolist()
            })
        return self.results

    def analyze(self,
                result_json_name: dict = None,
                use_package: bool = False,
                remove_mpf: bool = True,
                mpf_threshold: float = 1e-3,
                show: bool = False) -> dict:
        """
        Analyze results using Lomb-Scargle periodogram and identify significant critical times.

        Parameters
        ----------
        result_json_name : dict, optional
            Path to a JSON file containing results. If None, uses `self.results`.
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
        if result_json_name is None and self.results is None:
            raise ValueError("Results must be provided.")

        if result_json_name is not None:
            with open(result_json_name, "r") as f:
                self.results = json.load(f)

        self.best_results = []

        # Visualizations if requested
        if show:
            num_intervals = len(self.results)
            num_cols = 3
            num_rows = (num_intervals + num_cols - 1) // num_cols
            fig, axes = plt.subplots(3, 1, figsize=(12, 18))

        for idx, res in enumerate(tqdm(self.results, desc="Analyzing results", unit="result")):

            mask = (self.global_times >= res["sub_start"]) & (self.global_times <= res["sub_end"])
            t_sub = self.global_times[mask]
            y_sub = self.global_prices[mask]

            # Lomb-Scargle analysis
            lomb = LombAnalysis(self.lppl_model(t_sub, y_sub, res["bestParams"]))
            lomb.compute_lomb_periodogram(use_package=use_package)
            lomb.filter_results(remove_mpf=remove_mpf, mpf_threshold=mpf_threshold)
            is_significant = lomb.check_significance()

            if show and idx == 0 and self.stop == False:
                self.stop = True
                fig_residuals, ax_residuals = plt.subplots(figsize=(13, 6))
                lomb.show_residuals(ax=ax_residuals)
                ax_residuals.set_title('Subinterval Residuals')
                fig_residuals.savefig(f'residuals_{idx}.png')
                plt.close(fig_residuals)
            
                fig_spectrum, ax_spectrum = plt.subplots(figsize=(13, 6))
                lomb.show_spectrum(ax=ax_spectrum, use_filtered=False, show_threshold=True, highlight_freq=True)
                ax_spectrum.set_title('Subinterval Lomb-Scargle Spectrum')
                fig_spectrum.savefig(f'spectrum_{idx}.png')
                plt.close(fig_spectrum)

                fig_lppl, ax_lppl = plt.subplots(figsize=(13, 6))
                self.show_lppl(lomb.lppl, ax=ax_lppl)
                ax_lppl.set_title('Subinterval Log-Periodic Power Law Fit')
                fig_lppl.savefig(f'lppl_{idx}.png')
                plt.close(fig_lppl)



            self.best_results.append({
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

    def save_results(self, results: dict, file_name: str) -> None:
        """
        Save results to a JSON file.

        Parameters
        ----------
        results : dict
            Results to be saved.
        file_name : str
            Path to the output JSON file.
        """
        with open(file_name, "w") as f:
            json.dump(results, f, indent=4)

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

        length_extended = (round(lppl.tc) + 50) if self.frequency == "daily" else (round(lppl.tc) + 100) 

        # Calculate the maximum available length
        max_length = len(self.global_prices)

        # Adjust length_extended so it does not exceed the available length
        length_extended = min(length_extended, max_length)

        extended_t = np.arange(lppl.t[0], length_extended)
        extended_y = self.global_prices[int(extended_t[0]):int(extended_t[-1] + 1)]
        extended_dates = self.global_dates[int(extended_t[0]):int(extended_t[-1] + 1)]
        end_date = self.global_dates[int(lppl.t[-1])]

        lppl.t = extended_t
        full = lppl.predict(True)
        trend = lppl.predict(False)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(extended_dates, extended_y, label='Observed', color='black')
        ax.plot(extended_dates, full, label='Log Periodic Power Law Model', linestyle='--', color='black')
        ax.plot(extended_dates, trend, label='Power Law Model', linestyle='-.', color='darkgray')
        ax.axvline(x=end_date, color='r', linestyle=':', label='End of Subinterval')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.set_title('LPPL Model Prediction')
        ax.legend()

        if show:
            plt.show()

    def visualize(self) -> None:
        """
        Visualize significant critical times on the price series.
        """
        significant_tc = []
        min_time = np.inf
        max_time = -np.inf

        for res in self.best_results:
            if res["sub_start"] < min_time:
                min_time = res["sub_start"]
            if res["sub_end"] > max_time:
                max_time = res["sub_end"]
            if res["is_significant"]:
                significant_tc.append([res["bestParams"][0], res["power_value"]])

        plt.figure(figsize=(12, 6))
        plt.plot(self.global_dates, self.global_prices, label="Data", color="black")
        plt.axvline(x=self.global_dates[int(min_time)], color="gray", linestyle="--", label="Start Date")
        plt.axvline(x=self.global_dates[int(max_time)], color="gray", linestyle="--", label="End Date")


        # Garde les 15 plus significatifs
        significant_tc = sorted(significant_tc, key=lambda x: x[1], reverse=True)[:15]

        for tc, _ in significant_tc:
            if tc < (len(self.global_dates) - 1):
                plt.axvline(x=self.global_dates[int(round(tc))], color="red", linestyle=":")

        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.show()

    def generate_subintervals(self) -> None:
        """
        Generate subintervals based on the frequency and pseudo-code logic.
        """
        time_start = self.sample[0, 0]
        time_end = self.sample[-1, 0]

        if self.frequency == "daily":
            freq_list = [15, 30, 5]
        elif self.frequency == "weekly":
            freq_list = [3.0, 6.0, 1.0]
        elif self.frequency == "monthly":
            freq_list = [0.75, 1.5, 0.25]

        three_weeks, six_weeks, one_week = freq_list
        total_days = (time_end - time_start)
        delta = max((total_days * 0.75) / three_weeks, three_weeks)

        self.subintervals = []
        for sub_end in np.arange(time_end, time_end - six_weeks, -one_week):
            for sub_st in np.arange(time_start, time_end - total_days / 4, delta):
                mask = (self.sample[:, 0] >= sub_st) & (self.sample[:, 0] <= sub_end)
                sub_sample = self.sample[mask]
                if len(sub_sample) > 0:
                    self.subintervals.append((sub_st, sub_end, sub_sample))