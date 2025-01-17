import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from tqdm import tqdm
from datetime import datetime
import plotly.io as pio
import plotly.graph_objects as go
import os

from .Optimizers import MPGA, PSO, SGA, SA, Optimizer
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

    def __init__(self, frequency: str = "daily", input_type: InputType = InputType.WTI) -> None:
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
                data['Price'] = data['Price'].apply(lambda x: x / 8)  # Stock split 1:8 en 2020
                data["Date"] = pd.to_datetime(data["Date"], format="%d/%m/%Y").values.astype("datetime64[D]")

            case InputType.WTI:
                data = pd.read_csv(f'data/WTI_Spot_Price_{self.frequency}.csv', skiprows=4)
                data.columns = ["Date", "Price"]
                data["Date"] = pd.to_datetime(data["Date"], format="%m/%d/%Y").values.astype("datetime64[D]")

            case InputType.SP500:
                data = pd.read_csv(f'data/sp500_Price_daily.csv', sep=";")
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
                results: dict = None,
                result_json_name: str = None,
                lppl_model: 'LPPL | LPPLS' = LPPL,
                significativity_tc: float = 0.3,
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

    def visualise_data(self,
                       start_date: str = None,
                       end_date: str = None):
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
            print(f"Aucune donnée disponible entre {start_date} et {end_date}.")
            return

        filtered_dates = [self.global_dates[i] for i in filtered_indices]
        filtered_prices = [self.global_prices[i] for i in filtered_indices]
        fig = go.Figure()
        name = f"Evolution of {self.input_type.value} log price"
        fig.add_trace(
            go.Scatter(x=filtered_dates, y=np.log(filtered_prices), mode='lines', line=dict(color="black", width=1)))

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
                     best_results: dict,
                     name="",
                     data_name: str = "",
                     start_date: str = None,
                     end_date: str = None,
                     nb_tc: int = None,
                     real_tc: str = None) -> None:
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
        significant_tc = []
        min_time = np.inf
        max_time = -np.inf

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
            print(f"Aucune donnée disponible entre {start_date} et {end_date}.")
            return

        filtered_dates = [self.global_dates[i] for i in filtered_indices]
        filtered_prices = [self.global_prices[i] for i in filtered_indices]
        fig = go.Figure()
        # Plot de la série de prix
        fig.add_trace(go.Scatter(x=filtered_dates, y=filtered_prices, mode='lines', name=data_name,
                                 line=dict(color="black", width=1)))

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
            if (nb_tc != None):
                # Select the number of tc
                significant_tc = sorted(significant_tc, key=lambda x: x[1], reverse=True)[:nb_tc]
                significant_tc = [element[0] for element in significant_tc]

            else:
                significant_tc = [element[0] for element in significant_tc]
        except:
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
                            name=f"Critical times" if index_plot == 0 else None,
                            showlegend=(index_plot == 0)
                        )
                    )
                    index_plot += 1
            except:
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

    def visualize_compare_results(self, multiple_results: dict[str, dict],
                                  name: str = "",
                                  data_name: str = "",
                                  real_tc: str = None,
                                  optimiseurs_models: list = None,
                                  start_date: str = None,
                                  end_date: str = None,
                                  nb_tc: int = 20,
                                  save_plot: bool = False):
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
        start = start_date
        end = end_date

        name_plot = ""
        if start_date is not None and end_date is not None:
            start_date = pd.to_datetime(start_date, format="%d/%m/%Y")
            end_date = pd.to_datetime(end_date, format="%d/%m/%Y") + timedelta(days=10 * 365)
        else:
            start_date = np.min(self.global_dates)
            end_date = np.max(self.global_dates)
        # Filtration
        filtered_indices = [i for i, date in enumerate(self.global_dates) if start_date <= date <= end_date]
        if not filtered_indices:
            print(f"Aucune donnée disponible entre {start_date} et {end_date}.")
            return

        filtered_dates = [self.global_dates[i] for i in filtered_indices]
        filtered_prices = [self.global_prices[i] for i in filtered_indices]

        fig = go.Figure()
        # Plot de la série de prix
        fig.add_trace(go.Scatter(x=filtered_dates, y=filtered_prices, mode='lines', name=data_name,
                                 line=dict(color="black", width=1)))

        # Si la vraie date du tc est fournie, on la plot
        if real_tc is not None:
            target_date = pd.to_datetime(real_tc, format="%d/%m/%Y")

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

        # Je veux garder 1/5 du max de la time series en haut et en bas
        total_height = max(filtered_prices) - min(filtered_prices)
        base_y = total_height / 6
        remaining_height = total_height - 2 * base_y
        # On divise l'espace en restant pour que chaque model ait la même hauteur
        rectangle_height = remaining_height / len(multiple_results.keys())

        for i, (optimizer_name, results) in enumerate(multiple_results.items()):
            # Récupération du modèle LPPL correspondant
            lppl_model_name = optimiseurs_models[i] if optimiseurs_models and i < len(
                optimiseurs_models) else "Unknown Model"
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

            try:
                if (nb_tc != None):
                    significant_tc = sorted(significant_tc, key=lambda x: x[1], reverse=True)[
                                     :min(len(significant_tc), nb_tc)]
                # Calcul de la date exacte du tc en pondérant nb_tc par leur power
                sum_max_power = sum(x[1] for x in significant_tc if x[1] is not None and not np.isnan(x[1]))
                weighted_sum_tc = sum(x[0] * x[1] for x in significant_tc if x[1] is not None and not np.isnan(x[1]))
                significant_tc = weighted_sum_tc / sum_max_power if sum_max_power != 0 else 0
            except:
                pass

            # On plot les start et end date une fois à la première itération
            if i == 0:
                if start_date <= self.global_dates[int(min_time)] <= end_date:
                    fig.add_trace(go.Scatter(x=[self.global_dates[int(min_time)], self.global_dates[int(min_time)]],
                                             y=[min(filtered_prices), max(filtered_prices)], mode="lines",
                                             line=dict(color="gray", dash="dash"), name="Start Date", showlegend=True))

                if start_date <= self.global_dates[int(max_time)] <= end_date:
                    fig.add_trace(go.Scatter(x=[self.global_dates[int(max_time)], self.global_dates[int(max_time)]],
                                             y=[min(filtered_prices), max(filtered_prices)], mode="lines",
                                             line=dict(color="gray", dash="longdash"), name="End Date",
                                             showlegend=True))

            # Calcul des dates des tc
            if significant_tc and isinstance(significant_tc, float):
                print(f"Model : {optimizer_name}")
                if len(self.global_dates) > significant_tc > 0:
                    print(f"Significant TC : {self.global_dates[int(round(significant_tc))]}")
                    min_tc_date = self.global_dates[int(round(significant_tc))] - timedelta(days=15)
                    max_tc_date = self.global_dates[int(round(significant_tc))] + timedelta(days=15)
                elif significant_tc > len(self.global_dates):
                    extra_dates_needed = int(significant_tc) - len(self.global_dates) + 1
                    last_date = self.global_dates.max()
                    freq = "B" if self.frequency == "daily" else "W"
                    new_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=extra_dates_needed,
                                              freq=freq)

                    min_tc_date = new_dates[extra_dates_needed - 1] - timedelta(days=15)
                    max_tc_date = new_dates[extra_dates_needed - 1] + timedelta(days=15)

                else:
                    print("No significant TC found, or out of range")
                    continue

                # Rectangle pour le modèle
                fig.add_trace(go.Scatter(
                    x=[min_tc_date, max_tc_date, max_tc_date, min_tc_date, min_tc_date],
                    y=[min(filtered_prices) + base_y + i * (rectangle_height),
                       min(filtered_prices) + base_y + i * (rectangle_height),
                       min(filtered_prices) + base_y + i * (rectangle_height) + rectangle_height,
                       min(filtered_prices) + base_y + i * (rectangle_height) + rectangle_height,
                       min(filtered_prices) + base_y + i * (rectangle_height)],
                    fill="toself", fillcolor=colors[i % len(colors)], opacity=0.5, showlegend=True,
                    mode="lines+markers", marker=dict(size=1),
                    line=dict(color="gray", width=1), name=legend_label))

                # Ajout du nom du modèle au centre du rectangle
                center_x = min_tc_date + (max_tc_date - min_tc_date) / 2
                center_y = min(filtered_prices) + base_y + i * (rectangle_height) + rectangle_height / 2
                fig.add_trace(
                    go.Scatter(x=[center_x], y=[center_y], text=[optimizer_name], mode="text", showlegend=False))

        fig.update_layout(title=name,
                          xaxis=dict(
                              title='Date',  # Titre de l'axe X
                              showline=True,  # Afficher la ligne de l'axe X
                              linecolor='black',  # Couleur de la ligne de l'axe X
                              linewidth=1,  # Épaisseur de la ligne
                              mirror=True  # Ajouter la ligne de l'axe sur le côté opposé
                          ),
                          yaxis=dict(
                              title=f"{self.input_type.value} {self.frequency} price",  # Titre de l'axe Y
                              showline=True,  # Afficher la ligne de l'axe Y
                              linecolor='black',  # Couleur de la ligne de l'axe Y
                              linewidth=1,  # Épaisseur de la ligne
                              mirror=True  # Ajouter la ligne de l'axe sur le côté opposé
                          ),
                          showlegend=True,
                          plot_bgcolor='white',
                          paper_bgcolor='white')
        fig.show()
        if (save_plot):
            # Sauvegarde du plot
            start_date_obj = datetime.strptime(start, "%d/%m/%Y")
            end_date_obj = datetime.strptime(end, "%d/%m/%Y")
            filename = f"results_{self.input_type.value}/algo_comparison//{self.frequency}/{name_plot}{start_date_obj.strftime('%m-%Y')}_{end_date_obj.strftime('%m-%Y')}.png"
            self.save_image(fig, filename)

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

    @staticmethod
    def generate_subintervals(frequency: str, sample: np.asarray) -> list:
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
    def select_sample(data: np.asarray, time_start: str, time_end: str) -> np.ndarray:
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
            print(f"{directory_path} path was created !")
            os.makedirs(directory_path)

        with open(file_name, "w") as f:
            json.dump(results, f, indent=4)

    @staticmethod
    def save_image(fig, filename: str):
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
            print(f"{directory_path} path was created !")
            os.makedirs(directory_path)

        pio.write_image(fig, filename, scale=5, width=1000, height=800)