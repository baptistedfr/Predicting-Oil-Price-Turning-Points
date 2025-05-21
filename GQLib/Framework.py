import numpy as np
import pandas as pd
import json
import random
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from datetime import datetime
import plotly.io as pio
import plotly.graph_objects as go
import os
from GQLib.plotter import Plotter
from .Optimizers import Optimizer
from GQLib.LombAnalysis import LombAnalysis
from GQLib.Models import LPPL, LPPLS
from .enums import InputType
import logging
from GQLib.logging import with_spinner
import matplotlib.dates as mdates
from scipy.stats import gaussian_kde
import seaborn as sns


logger = logging.getLogger(__name__)

with open("params/debug_framework.json", "r") as file:
    debug_params = json.load(file)

DEBUG_STATUS_GRAPH_TC = debug_params["DEBUG_STATUS_GRAPH_TC"]
DEBUG_STATUS_GRAPH_LOMB = debug_params["DEBUG_STATUS_GRAPH_LOMB"]

class Framework:
    """
    Framework for processing and analyzing financial time series using LPPL and Lomb-Scargle techniques.

    This framework includes:
    - Data loading and subinterval generation.
    - Optimization of LPPL parameters using a custom optimizer.
    - Lomb-Scargle periodogram analysis for detecting significant frequencies.
    - Visualization of results, including LPPL predictions and significant critical times.
    """

    def __init__(self, frequency: str = "daily", input_type : InputType = InputType.WTI) -> None:
        """
        Initialize the Framework with a specified frequency for analysis.

        Parameters
        ----------
        frequency : str, optional
            The frequency of the time series data. Must be one of {"daily", "weekly"}.
            Default is "daily".

        input_type : InputType, optional
            The  input type of the data selected

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
        self.data = self.load_data()

        self.global_times = self.data[:, 0].astype(float)
        self.global_dates = self.data[:, 1]
        self.global_prices = self.data[:, 2].astype(float)

    @with_spinner("Loading data in progress ...")
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
                data = pd.read_csv('data/sp500_Price_daily.csv', sep=";")
                data.columns = ["Date", "Price"]
                data["Date"] = pd.to_datetime(data["Date"], format="%m/%d/%Y").values.astype("datetime64[D]")
            
            case InputType.BTC: 
                data = pd.read_csv(f'data/BTC_{self.frequency}.csv', sep=",")
                data.columns = ["Date", "Price"]
                data["Date"] = pd.to_datetime(data["Date"], format="%Y-%m-%d").values.astype("datetime64[D]")

            case InputType.SSE:
                data = pd.read_csv(f'data/SSE_Price_{self.frequency}.csv', sep=";")
                data.columns = ["Date", "Price"]
                data["Date"] = pd.to_datetime(data["Date"], format="%m/%d/%Y").values.astype("datetime64[D]")

            case InputType.EURUSD: 
                data = pd.read_csv(f'data/EURUSD_{self.frequency}.csv', sep=";")
                data.columns = ["Date", "Price"]
                data["Date"] = pd.to_datetime(data["Date"], format="%Y-%m-%d").values.astype("datetime64[D]")
            
        # Date conversion and sorting
        data = data.sort_values(by="Date")
        # Add numeric time index
        t = np.linspace(0, len(data) - 1, len(data))
        data = np.insert(data.to_numpy(), 0, t, axis=1)
        return data

    @with_spinner("Optimization with {optimizer.__class__.__name__} in progress ...")
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
        #for (sub_start, sub_end, sub_data) in tqdm(subintervals, desc="Processing subintervals", unit="subinterval"):
        for sub_start, sub_end, sub_data in subintervals:
            
            bestObjV, bestParams = optimizer.fit(sub_start, sub_end, sub_data)
            results.append({
                "sub_start": sub_start,
                "sub_end": sub_end,
                "bestObjV": bestObjV,
                "bestParams": bestParams.tolist()
            })
        return results

    @with_spinner("Lomb-Scargle analysis in progress ...")
    def analyze(self,
                results : dict = None,
                result_json_name: str = None,
                lppl_model: 'LPPL | LPPLS' = LPPL) -> dict:
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
        if DEBUG_STATUS_GRAPH_LOMB:
            num_intervals = len(results)
            num_cols = 3
            num_rows = (num_intervals + num_cols - 1) // num_cols
            fig, axes = plt.subplots(num_intervals, num_cols, figsize=(25, 18 * num_rows))

        for idx, res in enumerate(results):
            mask = (self.global_times >= res["sub_start"]) & (self.global_times <= res["sub_end"])
            t_sub = self.global_times[mask]
            y_sub = self.global_prices[mask]

            # Lomb-Scargle analysis
            lomb = LombAnalysis(lppl_model(t_sub, y_sub, res["bestParams"]))
            lomb.compute_lomb_periodogram()
            lomb.filter_results()
            is_significant = lomb.check_significance()

            if DEBUG_STATUS_GRAPH_LOMB:
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

        if DEBUG_STATUS_GRAPH_LOMB:
            pio.renderers.default = 'browser'
            plt.tight_layout()
            plt.show()

        return best_results

    def visualise_data(self,
                       start_date : str = None,
                       end_date : str = None):
        """
        Visualize the log price evolution over a specified date range

        Parameters:
            start_date (str, optional): The start date of the range to visualize, formatted as "DD/MM/YYYY". 
                                        Defaults to the earliest available date.
            end_date (str, optional): The end date of the range to visualize, formatted as "DD/MM/YYYY". 
                                    Defaults to the latest available date.
        """
        if start_date is not None:
            start_date = pd.to_datetime(start_date, format="%d/%m/%Y")
        else:
            start_date = self.global_dates.min()
        
        if end_date is not None:
            end_date = pd.to_datetime(end_date, format="%d/%m/%Y")
        else:
            end_date = self.global_dates.max()

        filtered_indices = [
            i for i, date in enumerate(self.global_dates) if start_date <= date <= end_date
        ]
        if not filtered_indices:
            logging.info(f"Aucune donnée disponible entre {start_date} et {end_date}.")
            return
        
        filtered_dates = [self.global_dates[i] for i in filtered_indices]
        filtered_prices = [self.global_prices[i] for i in filtered_indices]
        fig = go.Figure()
        name = f"Evolution of {self.input_type.value} log price"
        fig.add_trace(go.Scatter(x=filtered_dates, y=np.log(filtered_prices), mode='lines', line=dict(color="black", width=1)))

        fig.update_layout(title=name, 
                          xaxis=dict(
                            title='Date',               
                            showline=True,            
                            linecolor='black',         
                            linewidth=1,                
                            mirror=True                 
                        ),
                        yaxis=dict(
                            title=f"{self.input_type.value} {self.frequency} log price",              
                            showline=True,            
                            linecolor='black',          
                            linewidth=1,              
                            mirror=True                
                        ),
                          showlegend=False, 
                          plot_bgcolor='white', 
                          paper_bgcolor='white')
        fig.show()

    def visualize_tc(self, 
                  best_results : dict, 
                  name = "", 
                  data_name: str = "",
                  start_date: str = None, 
                  end_date: str = None, 
                  nb_tc : int = None,
                  real_tc : str = None) -> None:
        """
        Visualize significant critical times on the price series.
        Allows filtering and displaying results for a specific date range.
        
        Args:
            best_results (dict): Optimal results containing information about the turning points.
            name (str): Name of the graph.
            data_name (str) : Name of the data
            start_date (str): Start date (format: 'YYYY-MM-DD'). If None, uses the start of the data.
            end_date (str): End date (format: 'YYYY-MM-DD'). If None, uses the end of the data.
            nb_tc (int): Maximum number of turning points to display.
            real_tc (str): Actual value of the turning point.
        """

        logging.debug("\n Visualize function input :")
        logging.debug(f"best_results : {best_results}")
        logging.debug(f"name : {name}")
        logging.debug(f"data_name : {data_name}")
        logging.debug(f"start_date : {start_date}")
        logging.debug(f"end_date : {end_date}")
        logging.debug(f"nb_tc : {nb_tc}")
        logging.debug(f"real_tc : {real_tc}\n")
        

        significant_tc = []
        min_time = np.inf
        max_time = -np.inf

        logging.info("Visualisation des tc")

        if start_date is not None:
            start_date = pd.to_datetime(start_date, format="%d/%m/%Y")
        else:
            start_date = np.min(self.global_dates)
        
        if end_date is not None:
            end_date = pd.to_datetime(end_date, format="%d/%m/%Y")
        else:
            end_date = np.max(self.global_dates)

        filtered_indices = [
            i for i, date in enumerate(self.global_dates) if start_date <= date <= end_date
        ]
        if not filtered_indices:
            logging.info(f"Aucune donnée disponible entre {start_date} et {end_date}.")
            return
        
        filtered_dates = [self.global_dates[i] for i in filtered_indices]
        filtered_prices = [self.global_prices[i] for i in filtered_indices]
        fig = go.Figure()
        # Plot de la série de prix
        fig.add_trace(go.Scatter(x=filtered_dates, y=filtered_prices, mode='lines', name=data_name, line=dict(color="black", width=1)))

        # Si la vraie date du tc est fournie, on la plot
        if real_tc is not None:
            target_date = pd.to_datetime(real_tc, format="%d/%m/%Y")
    
            fig.add_trace(
                go.Scatter(
                    x=[target_date, target_date],
                    y=[min(filtered_prices), max(filtered_prices)],
                    mode="lines",
                    line=dict(color="green", width=4),
                    name="Real critical time",
                    showlegend=True
                )
            )

        for res in best_results:
            if res["sub_start"] < min_time:
                min_time = res["sub_start"]
            if res["sub_end"] > max_time:
                max_time = res["sub_end"]
            if res["is_significant"]:
                significant_tc.append([res["bestParams"][0], res["power_value"]])

        # Add of computing start date and end date 
        if start_date <= self.global_dates[int(min_time)] <= end_date:
            fig.add_trace(go.Scatter(x=[self.global_dates[int(min_time)], self.global_dates[int(min_time)]],
                                     y=[min(filtered_prices), max(filtered_prices)], mode="lines",
                                     line=dict(color="gray", dash="dash"), name="Start Date", showlegend=True))

        if start_date <= self.global_dates[int(max_time)] <= end_date:
            fig.add_trace(go.Scatter(x=[self.global_dates[int(max_time)], self.global_dates[int(max_time)]],
                                     y=[min(filtered_prices), max(filtered_prices)], mode="lines",
                                     line=dict(color="gray", dash="longdash"), name="End Date", showlegend=True))

        try:
            if (nb_tc is not None):
                # Select the number of tc
                significant_tc = sorted(significant_tc, key=lambda x: x[1], reverse=True)[:nb_tc]
                significant_tc = [element[0] for element in significant_tc]
                
            else:
                significant_tc = [element[0] for element in significant_tc]
        except Exception:
            pass
        
        index_plot = 0
        for tc in significant_tc:
            try:
                date_tc = self.global_dates[int(round(tc))]
                if start_date <= date_tc <= end_date:
                    fig.add_trace(
                        go.Scatter(
                            x=[date_tc, date_tc],
                            y=[min(filtered_prices), max(filtered_prices)],
                            mode="lines",
                            line=dict(color="red", dash="dot"),
                            name="Critical times" if index_plot == 0 else None,
                            showlegend=(index_plot == 0)
                        )
                    )
                    index_plot += 1
            except Exception:
                continue
        
        fig.update_layout(title=name, 
                          xaxis=dict(
                            title='Date',               
                            showline=True,             
                            linecolor='black',         
                            linewidth=1,               
                            mirror=True                 
                        ),
                        yaxis=dict(
                            title=f"{self.input_type.value} {self.frequency} price",              
                            showline=True,            
                            linecolor='black',          
                            linewidth=1,                
                            mirror=True                 
                        ),
                          showlegend=True, 
                          plot_bgcolor='white', 
                          paper_bgcolor='white')
        fig.show()

    @with_spinner("Creation of visualization in progress ...")
    def visualize_compare_results(self, multiple_results: dict[str, dict], 
                                  name: str = "", 
                                  data_name: str = "",
                                  real_tc: str = None, 
                                  optimiseurs_models : list = None,
                                  start_date: str = None, 
                                  end_date: str = None, 
                                  nb_tc: int = 20,
                                  save_plot : bool = False):
        """
        Visualize and compare multiple optimizers results on the same period
        Args:
            multiple_results (dict[str, dict]): dictionnary of results to display
            name (str, optional): Name of the graph Defaults to "".
            data_name (str, optional): name of the data. Defaults to "".
            real_tc (str, optional): The real tc to display.
            optimiseurs_models (list, optional): Optimizers Models .
            start_date (str, optional): start date of the computing interval. 
            end_date (str, optional): end date of the computing interval. Defaults to None.
            nb_tc (int, optional): Number of tc necessary to calcul the exact tc. Defaults to 20.
            save_plot (bool, optional): Whether to save the plot. Defaults to False.
        """

        # Adapt multiple_result to old format :
        temp = {}
        for key, value in multiple_results.items():
            temp[key] = value["raw_filtered_result"]
        multiple_results = temp

        logging.debug("\n Visualize function input :")
        logging.debug(f"multiple_results : {multiple_results}")
        logging.debug(f"name : {name}")
        logging.debug(f"data_name : {data_name}")
        logging.debug(f"real_tc : {real_tc}")
        logging.debug(f"optimiseurs_models : {optimiseurs_models}")
        logging.debug(f"start_date : {start_date}")
        logging.debug(f"end_date : {end_date}")
        logging.debug(f"nb_tc : {nb_tc}")
        logging.debug(f"save_plot : {save_plot}\n")

        colors = [
            "#ffa15a",  # Orange clair
            "#ab63fa",  # Violet clair
            "#00cc96",  # Vert clair
            "#ef553b",  # Rouge clair
            "#636efa",  # Bleu clair
            "#19d3f3",  # Cyan
            "#ff6692",  # Rose clair
            "#b6e880",  # Vert lime
            "#ff97ff",  # Magenta clair
        ]
        logging.debug("Starting visualize_compare_results with start_date=%s and end_date=%s", start_date, end_date)
        start = start_date
        end = end_date

        name_plot = ""
        if start_date is not None and end_date is not None:
            try:
                start_date = pd.to_datetime(start_date, format="%d/%m/%Y")
                end_date = pd.to_datetime(end_date, format="%d/%m/%Y") + timedelta(days=365)
                logging.debug("Parsed start_date: %s, end_date: %s", start_date, end_date)
            except Exception as e:
                logging.error("Error parsing start_date or end_date: %s", e)
                raise
        else:
            start_date = np.min(self.global_dates)
            end_date = np.max(self.global_dates)
            logging.warning("start_date or end_date is None. Using global date range: %s to %s", start_date, end_date)
        # Filtration
        filtered_indices = [i for i, date in enumerate(self.global_dates) if start_date <= date <= end_date]
        if not filtered_indices:
            logging.info("Aucune donnée disponible entre %s et %s.", start_date, end_date)
            return

        filtered_dates = [self.global_dates[i] for i in filtered_indices]
        filtered_prices = [self.global_prices[i] for i in filtered_indices]
        logging.debug("Filtered %d data points for visualization", len(filtered_dates))

        fig = go.Figure()
        # Plot de la série de prix
        fig.add_trace(go.Scatter(x=filtered_dates, y=filtered_prices, mode='lines', name=data_name, line=dict(color="black", width=1)))
        logging.debug("Base price series plotted")

        # Si la vraie date du tc est fournie, on la plot
        if real_tc is not None:
            try:
                target_date = pd.to_datetime(real_tc, format="%d/%m/%Y")
                logging.debug("Parsed real critical time: %s", target_date)
            except Exception as e:
                logging.error("Error parsing real_tc: %s", e)
            target_date = None
            if target_date:
                fig.add_trace(
                    go.Scatter(
                    x=[target_date, target_date],
                    y=[min(filtered_prices), max(filtered_prices)],
                    mode="lines",
                    line=dict(color="red", width=4),
                    name="Real critical time",
                    showlegend=True
                    )
                )
            logging.debug("Real critical time plotted at %s", target_date)

        # Je veux garder 1/5 du max de la time series en haut et en bas
        total_height = max(filtered_prices) - min(filtered_prices)
        base_y = total_height / 6
        remaining_height = total_height - 2 * base_y
        # On divise l'espace restant pour que chaque modèle ait la même hauteur
        rectangle_height = remaining_height / len(multiple_results.keys())
        logging.debug("Calculated rectangle_height: %s", rectangle_height)

        for i, (optimizer_name, results) in enumerate(multiple_results.items()):
            logging.debug("Processing optimizer: %s", optimizer_name)
            # Récupération du modèle LPPL correspondant
            lppl_model_name = optimiseurs_models[i] if optimiseurs_models and i < len(optimiseurs_models) else "Unknown Model"
            legend_label = f"{optimizer_name} ({lppl_model_name})"
            name_plot += f"{optimizer_name}({lppl_model_name})_"
            best_results = results
            significant_tc = []
            min_time = np.inf
            max_time = -np.inf

            for res in best_results:
                if res["sub_start"] < min_time:
                    min_time = res["sub_start"]
                if res["sub_end"] > max_time:
                    max_time = res["sub_end"]
                if res["is_significant"]:
                    significant_tc.append([res["bestParams"][0], res["power_value"]])
            logging.debug("Optimizer '%s': min_time=%s, max_time=%s, significant_tc=%s", optimizer_name, min_time, max_time, significant_tc)
            
            try:
                if (nb_tc is not None):
                    significant_tc = sorted(significant_tc, key=lambda x: x[1], reverse=True)[:min(len(significant_tc), nb_tc)]
                    logging.debug("Trimmed significant_tc: %s", significant_tc)
                # Calcul de la date exacte du tc en pondérant nb_tc par leur power
                sum_max_power = sum(x[1] for x in significant_tc if x[1] is not None and not np.isnan(x[1]))
                weighted_sum_tc = sum(x[0] * x[1] for x in significant_tc if x[1] is not None and not np.isnan(x[1]))
                significant_tc = weighted_sum_tc / sum_max_power if sum_max_power != 0 else 0
                logging.debug("Computed weighted significant tc: %s", significant_tc)
            except Exception as e:
                logging.error("Error processing significant_tc for optimizer '%s': %s", optimizer_name, e)
                continue

            # On plot les start et end date une fois à la première itération
            if i == 0:
                if start_date <= self.global_dates[int(min_time)] <= end_date:
                    fig.add_trace(go.Scatter(x=[self.global_dates[int(min_time)], self.global_dates[int(min_time)]],
                        y=[min(filtered_prices), max(filtered_prices)], mode="lines",
                        line=dict(color="gray", dash="dash"), name="Start Date", showlegend=True))
                    logging.debug("Start Date plotted")
                if start_date <= self.global_dates[int(max_time)] <= end_date:
                    fig.add_trace(go.Scatter(x=[self.global_dates[int(max_time)], self.global_dates[int(max_time)]],
                            y=[min(filtered_prices), max(filtered_prices)], mode="lines",
                            line=dict(color="gray", dash="longdash"), name="End Date", showlegend=True))
                    logging.debug("End Date plotted")
                
            # Calcul des dates des tc
            if significant_tc and isinstance(significant_tc, float):
                logging.info("Model '%s'", optimizer_name)
                if len(self.global_dates) > significant_tc > 0:
                    logging.info("Significant TC : %s", self.global_dates[int(round(significant_tc))])
                    min_tc_date = self.global_dates[int(round(significant_tc))] - timedelta(days=15)
                    max_tc_date = self.global_dates[int(round(significant_tc))] + timedelta(days=15)
                elif significant_tc > len(self.global_dates):
                    extra_dates_needed = int(significant_tc) - len(self.global_dates) + 1
                    last_date = self.global_dates.max()
                    freq = "B" if self.frequency == "daily" else "W"
                    new_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=extra_dates_needed, freq=freq)
                    min_tc_date = new_dates[extra_dates_needed - 1] - timedelta(days=15)
                    max_tc_date = new_dates[extra_dates_needed - 1] + timedelta(days=15)
                    logging.debug("New dates created for tc beyond range")
                else:
                    logging.info("No significant TC found, or out of range for optimizer '%s'", optimizer_name)
                    continue

                # Rectangle pour le modèle
                fig.add_trace(go.Scatter(
                    x=[min_tc_date, max_tc_date, max_tc_date, min_tc_date, min_tc_date],
                    y=[min(filtered_prices) + base_y + i * rectangle_height, 
                    min(filtered_prices) + base_y + i * rectangle_height,
                    min(filtered_prices) + base_y + i * rectangle_height + rectangle_height,
                    min(filtered_prices) + base_y + i * rectangle_height + rectangle_height, 
                    min(filtered_prices) + base_y + i * rectangle_height],
                    fill="toself", fillcolor=colors[i % len(colors)], opacity=0.5, showlegend=True,
                    mode="lines+markers", marker=dict(size=1), 
                    line=dict(color="gray", width=1), name=legend_label))
                logging.debug("Rectangle plotted for optimizer '%s'", optimizer_name)

                # Ajout du nom du modèle au centre du rectangle
                center_x = min_tc_date + (max_tc_date - min_tc_date) / 2
                center_y = min(filtered_prices) + base_y + i * rectangle_height + rectangle_height / 2
                fig.add_trace(go.Scatter(x=[center_x], y=[center_y], text=[optimizer_name], mode="text", showlegend=False))
                logging.debug("Label added at center for optimizer '%s'", optimizer_name)

                fig.update_layout(title=name, 
                        xaxis=dict(
                            title='Date',               
                            showline=True,            
                            linecolor='black',         
                            linewidth=1,                
                            mirror=True                 
                        ),
                        yaxis=dict(
                            title=f"{self.input_type.value} {self.frequency} price",              
                            showline=True,            
                            linecolor='black',          
                            linewidth=1,                
                            mirror=True                 
                        ),
                        showlegend=True, 
                        plot_bgcolor='white', 
                        paper_bgcolor='white')
        pio.renderers.default = 'browser'
        fig.show()
        logging.info("Figure displayed")
        if save_plot:
            try:
                start_date_obj = datetime.strptime(start, "%d/%m/%Y")
                end_date_obj = datetime.strptime(end, "%d/%m/%Y")
                filename = f"results_{self.input_type.value}/algo_comparison//{self.frequency}/{name_plot}{start_date_obj.strftime('%m-%Y')}_{end_date_obj.strftime('%m-%Y')}.png"
                self.save_image(fig, filename)
                logging.info("Figure saved at %s", filename)
            except Exception as e:
                logging.error("Error saving figure: %s", e)

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
        length_extended = (round(lppl.tc) + 200) if self.frequency == "daily" else (round(lppl.tc) + 100) 

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
        without_osc_predicted = lppl.predict(False)

        if ax is None:
            fig, ax = plt.subplots(figsize=(15, 9))

        ax.plot(extended_dates, extended_y, label='Observed', color='black', linewidth=1)
        ax.plot(extended_dates, predicted, label='Log Periodic Power Law', color='blue', linewidth=3, alpha=0.8)
        ax.plot(extended_dates, without_osc_predicted, label='Power Law', color='red', linewidth=3, alpha=0.8)
        ax.axvline(x=end_date, color='black', linestyle='--', label='End of Subinterval')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.set_title('LPPL Model Prediction')
        ax.legend()

        if show:
            plt.show()
        #     plotter = Plotter()
        #     plotter.plot_lppl_fit(lppl, self.global_dates, self.global_prices)
    
    @staticmethod
    def generate_subintervals(frequency :str, sample : np.asarray) -> list:
        """
        Generate subintervals based on the frequency and pseudo-code logic.
        Parameters
        ----------
        frequency : str
            The frequency of analysis, e.g., 'daily', 'weekly', or 'monthly'.
        sample : np.ndarray
            The dataset for a specific sample with columns: time index and price.

        Returns
        -------
        list
            A list of tuples representing subintervals. Each tuple contains:
            - Start time of the subinterval (float).
            - End time of the subinterval (float).
            - Sub-sample data (np.ndarray) within the interval.
        """
        time_start = sample[0, 0]
        time_end = sample[-1, 0]

        if frequency == "daily":
            freq_list = [15, 30, 5]
        elif frequency == "weekly":
            freq_list = [3.0, 6.0, 1.0]
        elif frequency == "monthly":
            freq_list = [0.75, 1.5, 0.25]

        three_weeks, six_weeks, one_week = freq_list
        total_days = (time_end - time_start)
        delta = max((total_days * 0.75) / three_weeks, three_weeks)

        subintervals = []
        for sub_end in np.arange(time_end, time_end - six_weeks, -one_week):
            for sub_st in np.arange(time_start, time_end - total_days / 4, delta):
                mask = (sample[:, 0] >= sub_st) & (sample[:, 0] <= sub_end)
                sub_sample = sample[mask]
                if len(sub_sample) > 0:
                    subintervals.append((sub_st, sub_end, sub_sample))
        return subintervals

    @staticmethod
    def select_sample(data : np.asarray, time_start: str, time_end: str) -> np.ndarray:
        """
        Select a sample from the global time series based on a user-defined date range.

        Parameters
        ----------
        data : np.ndarray
            The global dataset as a NumPy array with columns: time index, date, and price.
        time_start : str
            The start date for the selection in the format "%d/%m/%Y".
        time_end : str
            The end date for the selection in the format "%d/%m/%Y".
        Returns
        -------
        np.ndarray
            A 2D array of shape (M, 2), where:
            - Column 0: Numeric time indices (float).
            - Column 1: Prices (float).
        """
        # Convert start and end dates to datetime64
        start_dt = np.datetime64(pd.to_datetime(time_start, format="%d/%m/%Y"))
        end_dt = np.datetime64(pd.to_datetime(time_end, format="%d/%m/%Y"))

        # Filter rows within the specified date range
        mask = (data[:, 1] >= start_dt) & (data[:, 1] <= end_dt)
        sample = data[mask]

        return sample[:, [0, 2]].astype(float)

    @staticmethod
    def save_results(results: dict, file_name: str) -> None:
        """
        Save results to a JSON file.

        Parameters
        ----------
        results : dict
            Results to be saved.
        file_name : str
            Path to the output JSON file.
        """
        directory_path = os.path.dirname(file_name)

        if not os.path.exists(directory_path):
            logging.info(f"{directory_path} path was created !")
            os.makedirs(directory_path)


        with open(file_name, "w") as f:
            json.dump(results, f, indent=4)
    
    @staticmethod
    def save_image(fig , filename : str):
        """
        Save image to a png file.

        Parameters
        ----------
        fig : Figure
            Figure to be saved.
        filename : str
            Path to the output png file.
        """
        directory_path = os.path.dirname(filename)

        if not os.path.exists(directory_path):
            logging.info(f"{directory_path} path was created !")
            os.makedirs(directory_path)

        pio.write_image(fig, filename, scale=5, width=1000, height=800)

    def _base(self, start_training: str, end_date: str, real_tc: str = None, title: str = None, width: int = 18, height: int = 8) -> tuple:
        """
        Trace le prix + t1, t2, real_tc et retourne fig, ax
        avec zorder élevés par défaut.
        """
        # conversion des dates
        start_training = pd.to_datetime(start_training, format="%d/%m/%Y")
        end_training   = pd.to_datetime(end_date,      format="%d/%m/%Y")

        # fenêtre
        window_start = start_training - timedelta(days=90)
        window_end   = end_training   + timedelta(days=730)

        # extraction
        mask   = [(window_start <= d <= window_end) for d in self.global_dates]
        dates  = [d for d, m in zip(self.global_dates, mask) if m]
        prices = [p for p, m in zip(self.global_prices, mask) if m]

        # création figure/axe
        fig, ax = plt.subplots(figsize=(width, height))

        # rendre la figure et l'axe transparents
        fig.patch.set_alpha(0)     # fond de la figure
        ax.patch.set_alpha(0)      # fond de l'axe

        # prix
        ax.plot(dates, prices,
                color="black", linewidth=1.2,
                label=f"{self.input_type.value} {self.frequency} price",
                zorder=20)

        # bornes t1/t2
        ax.axvline(x=start_training, color="black",
                linestyle="-.", linewidth=1, label="t1",
                zorder=21)
        ax.axvline(x=end_training,   color="black",
                linestyle="-.", linewidth=1, label="t2",
                zorder=21)
        # fill between the all space between t1 and t2
        ax.axvspan(start_training, end_training, facecolor="gray", alpha=0.15)

        # real_tc
        if real_tc is not None:
            if isinstance(real_tc, str):
                real_tc = pd.to_datetime(real_tc, format="%d/%m/%Y")
            elif isinstance(real_tc, int):
                real_tc = self.global_dates[real_tc]
            ax.axvline(x=real_tc, color="red",
                    linewidth=2, label="Real critical time",
                    zorder=22)

        if title is not None:
            ax.set_title(title)
        else:
            ax.set_title("Log-Price & LPPLS fits")
            
        ax.set_xlabel("Date")
        ax.set_ylabel(f"{self.input_type.value} {self.frequency} price")
        leg = ax.legend(loc="upper left", title="Algorithmes / bornes")
        leg.get_frame().set_alpha(0.9)
        # Puis on remonte la légende
        leg.set_zorder(25)

        plt.tight_layout()
        return fig, ax

    def _add_tc(self, fig, ax, dict_results, algorithm: str = "SA"):
        list_tc = dict_results[algorithm]['tc_distrib']
        list_tc = [int(round(tc)) for tc in list_tc]
        list_tc = [self.global_dates[i] for i in list_tc]

        for idx, tc in enumerate(list_tc):
            if idx == 0:
                label = "Critical times"
            else:
                label = None
            ax.axvline(x=tc, color="green", linewidth=1, linestyle="--",
                    label=label, zorder=22)

        ax.legend(title="Critical times distributions", loc="upper left", fontsize=10)
        return fig, ax


    def _add_half_violins(self, fig, ax, dict_results,
                        width_scale: float = 0.5,
                        spacing: float = 0.5,
                        specific: str = "tc_distrib",
                        hatch_pattern: str = "/",
                        color: str = "white",
                        text: bool = False,
                        limit: str = None):
        """
        Ajoute des demi-violons hachurés, en coupant tout ce qui est
        antérieur à end_date (si fourni), et sans planter sur tableaux vides.
        """

        # 1) calcul de la limite en float Matplotlib
        if limit is not None:
            limit_dt  = pd.to_datetime(limit, format="%d/%m/%Y")
            # on cherche l'index exact dans global_dates
            idxs = np.where(np.array(self.global_dates) == limit_dt)[0]
            limit_num = mdates.date2num(self.global_dates[idxs[0]]) if len(idxs) else None
        else:
            limit_num = None

        # 2) axe secondaire sous l'axe principal
        ax_v = ax.twinx()
        ax_v.set_zorder(0)
        ax.set_zorder(1)
        ax_v.patch.set_alpha(0)
        ax_v.set_yticks([])
        ax_v.set_ylabel("")

        # 3) normalisation de self.global_dates en datetime
        global_dates = [
            pd.to_datetime(d, format="%d/%m/%Y") if isinstance(d, str) else d
            for d in self.global_dates
        ]

        # 4) collecte uniquement des distributions non-vides
        valid = {}
        for opt, vals in dict_results.items():
            raw = vals.get(specific, [])
            if not raw:
                valid[opt] = None
                continue
            idxs = [int(round(i)) for i in raw]
            # si un indice dépasse, on l'ajuste à la dernière date dispo
            idxs = [min(i, len(global_dates)-1) for i in idxs]
            dates = [global_dates[i] for i in idxs]
            nums  = mdates.date2num(dates)
            if nums.size > 0:
                valid[opt] = nums
            else:
                valid[opt] = nums

        # si aucune distribution valide, on sort sans rien tracer
        if not valid:
            return fig, ax

        # 5) calcul de la grille de dates sur laquelle on fera la KDE
        mn = min(nums.min() for nums in valid.values() if nums is not None)
        mx = max(nums.max() for nums in valid.values() if nums is not None)
        full_grid = np.linspace(mn, mx, 200)

        # 6) on ne garde que la partie >= limit_num si défini
        if limit_num is not None:
            date_grid = full_grid[full_grid >= limit_num]
        else:
            date_grid = full_grid

        # s'il n'y a plus rien après filtrage
        if date_grid.size == 0:
            return fig, ax

        # 7) tracé des demi-violons
        max_y = -np.inf

        print(len(valid))
        for idx, (opt, nums) in enumerate(valid.items()):
            
            print(opt)
            if nums is None:
                kde = gaussian_kde(random.sample(range(int(round(mn)), int(round(mx))), 100))
                alpha = 0.0
            elif len(nums) < 5:
                print("BIM")
                kde = gaussian_kde(random.sample(range(int(round(mn)), int(round(mx))), 100))
                alpha = 0.0
            else:
                kde  = gaussian_kde(nums)
                alpha = 0.9


            dens = kde(date_grid)
            # mise à l'échelle
            dens = dens / dens.max() * width_scale
            y0   = idx * spacing

            # base du violon (commence à limit_num sinon à mn)
            xmin = limit_num if limit_num is not None else mn
            ax_v.hlines(y=y0, xmin=xmin, xmax=mx,
                        colors="black", linewidth=0.5)

            # demi-violon blanc + hachures
            poly = ax_v.fill_between(
                date_grid, y0, y0 + dens,
                facecolor=color, edgecolor="black",
                linewidth=0.8, alpha=alpha
            )
            poly.set_hatch(hatch_pattern)

            # contour supérieur
            ax_v.plot(date_grid, y0 + dens,
                    color="black", linewidth=1.0, alpha=alpha)

            # label optionnel
            if text:
                ax_v.text(mx + (mx-xmin)*0.01,
                        y0 + dens.max()*0.5,
                        opt.replace("_", "\n"),
                        va="center", ha="left")

            max_y = max(max_y, y0 + dens.max())

        # 8) finalisation
        ax_v.set_ylim(-spacing*0.5, max_y + spacing*0.5)
        ax_v.xaxis_date()
        fig.autofmt_xdate()
        return fig, ax

    def _add_lppl_fit(self, fig, ax, dict_results: dict, nb_calib: int = 3, window_extension: int = 300, subintervals: bool = False):
        """
        Ajoute les courbes de fit LPPL/LPPLS et leurs intervalles en bas du graphique,
        triés par longueur de sous-intervalle et espacés verticalement.
        """
        calib_set = dict_results["NELDER_MEAD"]["raw_run_result"]
        # Échantillonnage aléatoire
        indices = random.sample(range(len(calib_set)), nb_calib)
        selected = [calib_set[i] for i in indices]

        # Constituer et trier la liste des intervalles (longueur décroissante)
        intervals = []  # (length, start_idx, end_idx, info)
        for info in selected:
            # convertir en int sûr
            start_idx = int(round(info["sub_start"]))
            end_idx = int(round(info["sub_end"]))
            length = end_idx - start_idx
            intervals.append((length, start_idx, end_idx, info))
        intervals.sort(key=lambda x: (-x[0], x[1]))

        # Calcul des positions verticales pour les hlines
        ymin, ymax = ax.get_ylim()
        total_h = ymax - ymin
        band_h = 0.20* total_h
        y_base = ymin + 0.05 * total_h

        colors = sns.dark_palette("navy", n_colors=len(intervals), reverse=False)

        for idx, (length, start_idx, end_idx, info) in enumerate(intervals):

            mask_cal = [(start_idx <= t <= end_idx) for t in self.global_times]
            t_cal = [d for d, m in zip(self.global_times, mask_cal) if m]
            prices_cal = [p for p, m in zip(self.global_prices, mask_cal) if m]

            model = LPPLS(params=info["bestParams"], t=np.array(t_cal), y=np.array(prices_cal))

            mask_ext = [(start_idx <= t <= end_idx + window_extension) for t in self.global_times]
            dates_ext = [d for d, m in zip(self.global_dates, mask_ext) if m]
            model.t = np.array([t for t, m in zip(self.global_times, mask_ext) if m])

            y_pred = model.predict(include_oscillation=True)
            ax.plot(
                dates_ext,
                y_pred,
                linestyle="--",
                linewidth=2,
                color=colors[idx % len(colors)],
                label=self.global_dates[int(round(model.tc))].strftime('%d/%m/%Y')
            )

            if subintervals:
                # Tracer les intervalles de calibration sur l'axe principal
                start_idx_clamped = max(start_idx, 0)
                end_idx_clamped = min(end_idx, len(self.global_dates) - 1)
                start_date_val = self.global_dates[start_idx_clamped]
                end_date_val = self.global_dates[end_idx_clamped]
                y_pos = y_base + 2 * idx #* (band_h / max(1, nb_calib - 1))
                ax.hlines(
                    y=y_pos,
                    xmin=start_date_val,
                    xmax=end_date_val,
                    colors="black",
                    linewidth=4,
                    alpha=0.8
                )

        ax.legend(title="Fits LPPL & Intervalles" if subintervals else "Fits LPPL", loc="upper left", fontsize=10)
        fig.autofmt_xdate()

        return fig, ax