from abc import ABC, abstractmethod
from typing import List, Tuple, Union
import pandas as pd
import numpy as np
from enum import Enum


class SubIntervalsSelection(ABC):

    def __init__(self, time_series: Tuple[np.ndarray, pd.Series]):
        """
        Parameters:
            time_series (Tuple[np.ndarray, pd.Series]): 
                - Numeric time index
                - Time series data
        """
        self.time_series = time_series

    @abstractmethod
    def get_subintervals(self) -> List[Tuple[float, float, pd.Series]]:
        """
        Get the subintervals of the time series.

        Returns:
            List[Tuple[float, float, np.ndarray]]: set of subintervals
                - start date as numeric value
                - end date as numeric value
                - time series data
        """
        pass

class MiaouIntervals(SubIntervalsSelection):

    def __init__(self, time_series: np.ndarray):
        """
        Parameters:
            time_series (np.ndarray): 
                - Numeric time index
                - Datetime 
                - Time series data
        """
        self.time_series = time_series

    def get_subintervals(self) -> List[Tuple[float, float, np.ndarray]]:
        """
        Get the subintervals of the time series.

        Returns:
            List[Tuple[float, float, np.ndarray]]: set of subintervals
                - start date as numeric value
                - end date as numeric value
                - time series data
        """
        time_start = self.time_series[0, 0]
        time_end = self.time_series[-1, 0]

        three_weeks, six_weeks, one_week = [15, 30, 5]
        total_days = (time_end - time_start)
        delta = max((total_days * 0.75) / three_weeks, three_weeks)

        subintervals = []
        for sub_end in np.arange(time_end, time_end - six_weeks, -one_week):
            for sub_st in np.arange(time_start, time_end - total_days / 4, delta):
                mask = (self.time_series[:, 0] >= sub_st) & (self.time_series[:, 0] <= sub_end)
                sub_sample = self.time_series[mask]
                if len(sub_sample) > 0:
                    subintervals.append((sub_st, sub_end, sub_sample))

        return subintervals
    
class DidouIntervals(SubIntervalsSelection):

    def __init__(self, 
                 time_series: Tuple[np.ndarray, pd.Series], 
                 minimal_length: int = 125):
        """
        Parameters:
            time_series (np.ndarray): 
                - Numeric time index
                - Datetime 
                - Time series data
        """
        self.time_series = time_series
        self.minimal_length = minimal_length

    def get_subintervals(self) -> List[Tuple[float, float, np.ndarray]]:
        """
        Get the subintervals of the time series with Sornette method.
        We start by getting the whole time series and we shift the start date in steps of 5 days.

        Returns:
            List[Tuple[float, float, np.ndarray]]: set of subintervals
                - start date as numeric value
                - end date as numeric value
                - time series data
        """
        time_start = self.time_series[0, 0]
        time_end = self.time_series[-1, 0]

        subintervals = []
        for sub_st in np.arange(time_start, time_end - self.minimal_length, 5):
            sub_end = time_end
            mask = (self.time_series[:, 0] >= sub_st) & (self.time_series[:, 0] <= sub_end)
            sub_sample = self.time_series[mask]
            subintervals.append((sub_st, sub_end, sub_sample))

        return subintervals
    
class SubIntervalMethod(Enum):
    MIAOU = MiaouIntervals
    DIDOU = DidouIntervals
