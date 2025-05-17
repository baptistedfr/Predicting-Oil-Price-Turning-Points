from typing import List, Dict, Any
import pandas as pd
import numpy as np
from .Optimizers import Optimizer
from .subintervals import MiaouIntervals, DidouIntervals, SubIntervalMethod
from .filterings import AbstractFilter
from typing import List, Tuple, Union
from enum import Enum

class DataName(Enum):
    BTC = "BTC_daily.csv"
    EUR_USD = "EURUSD_daily.csv"
    CS300 = "CSI300_Price_daily.csv"
    SP500 = "sp500_Price_daily.csv"
    USO = "USO_daily.csv"
    WTI = "WTI_Spot_Price_daily.csv"
    SSE = "SSE_Price_daily.csv"

class MiaouFramework:

    def __init__(self, 
                 data_names: List[DataName],
                 set_dates: Dict[str, Tuple[str, str]],
                 frequency: int,
                 optimizer: Optimizer,
                 filtering_method: AbstractFilter,
                 subinterval_method: SubIntervalMethod,
                 window_lenght: int):
        """
        Parameters:
            data_names (DataName): reference to the data name
            set_dates (Dict[str, List[str, str]]): set of dates ("Period 1": ["2020-01-01", "2020-12-31"], etc.)
            frequency (int): frequency of crash probability calculation in days (1: daily, 5: weekly, 30: monthly, etc.)
            optimizer (Optimizer): optimization method to use (Nelder-Mead, MPGA, etc.)
            filtering_method (AbstractFilter): filtering method to use (Lomb, Bounds, etc.)
            subinterval_method (SubIntervalMethod): subinterval method to use (shrinking, classic, etc.)
            window_lenght (int): length of the series to use for the crash probability calculation
        """
        self.data_names = data_names
        self.set_dates = set_dates
        self.frequency = frequency
        self.optimizer = optimizer
        self.filtering_method = filtering_method()
        self.subinterval_method = subinterval_method
        self.window_lenght = window_lenght

        self.optimizer.configure_params_from_frequency("daily", self.optimizer.__class__.__name__)

    def run(self):

        self.results_dict = {}

        for data in self.data_names:

            self.results_dict[data.name] = {}

            print(f"Running the analysis for {data.name} ...")
            print(f"Loading the data from {data.value} ...")

            time_series = self._load_data(data.value)

            timestamp = time_series[:, 1]

            for period, dates_tuple in self.set_dates[str(data.name)].items():

                self.results_dict[data.name][period] = {}
                self.results_dict[data.name][period]["confidence"] = pd.Series(dtype=float)
                self.results_dict[data.name][period]["price"] = pd.Series(dtype=float)
                
                start_date, end_date = dates_tuple
                print(f"Calculating the crash probability of {period} from {start_date} to {end_date} ...")

                start_date_dt = pd.to_datetime(start_date)
                end_date_dt = pd.to_datetime(end_date)

                start_idx = pd.DatetimeIndex(timestamp).get_indexer([start_date_dt], method='ffill')[0]
                end_idx = pd.DatetimeIndex(timestamp).get_indexer([end_date_dt], method='ffill')[0]
                
                # Select the time series  as : [t1 - window_lenght, t2]
                sub_series = time_series[start_idx - self.window_lenght:end_idx]

                for i in range(0, len(sub_series) - self.window_lenght, self.frequency):
                    
                    
                    window = sub_series[i:i + self.window_lenght]

                    confidence = self._compute_crash_proba(window[:, [0, 2]].astype(float))
                    price = sub_series[i + self.window_lenght, 2]
                    date = sub_series[i + self.window_lenght, 1]

                    self.results_dict[data.name][period]["confidence"].at[date] = confidence
                    self.results_dict[data.name][period]["price"].at[date] = price

    def _load_data(self, file_path: str) -> np.ndarray:
        """
        Load the time series data from a CSV file.
        The CSV file should contain a "Date" column and a "Price" column.

        Parameters:
            file_path (str): path to the CSV file

        Returns:
            np.ndarray: 
                - Numeric time index
                - Datetime 
                - Time series data
        """
        df = pd.read_csv(f'data/{file_path}', sep=",")
        df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d").values.astype("datetime64[D]")
        df = df.sort_values(by="Date")

        t = np.linspace(0, len(df) - 1, len(df))
        df = np.insert(df.to_numpy(), 0, t, axis=1)

        return df
    
    def _compute_crash_proba(self, time_series: np.ndarray) -> float:
        """
        Compute the crash probability of the time series.
        The crash probability is defined as : number of validated subintervals / number of subintervals 

        Parameters:
            time_series (np.ndarray): 
                - Numeric time index
                - Datetime 
                - Time series data

        Returns:
            float: crash probability
        """
        sub_intervals = self.subinterval_method.value(time_series).get_subintervals()

        crash_proba = 0
        for sub_start, sub_end, sub_data in sub_intervals:
            _, bestParams = self.optimizer.fit(sub_start, sub_end, sub_data)

            model = self.optimizer.lppl_model(sub_data[:, 0], sub_data[:, 1], bestParams)
            linear_params, non_linear_params = model.get_linear_params(), model.get_non_linear_params()

            if self.filtering_method.is_valid(linear_params, non_linear_params):
                crash_proba += 1

        return crash_proba / len(sub_intervals)
    
    def visualize(self):
        import matplotlib.pyplot as plt

        for asset, asset_data in self.results_dict.items():
            for period, period_data in asset_data.items():
                fig, ax1 = plt.subplots(figsize=(10, 6))

                dates = period_data["confidence"].index

                ax1.set_xlabel("Date")
                ax1.set_ylabel("Confidence", color="tab:blue")
                ax1.plot(dates, period_data["confidence"].values, color="tab:blue", label="Confidence")
                ax1.tick_params(axis="y", labelcolor="tab:blue")

                ax2 = ax1.twinx()
                ax2.set_ylabel("Price", color="tab:red")
                ax2.plot(dates, period_data["price"].values, color="tab:red", label="Price")
                ax2.tick_params(axis="y", labelcolor="tab:red")

                plt.title(f"{asset} - {period}")
                fig.tight_layout()
                plt.show()
                        