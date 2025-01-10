import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from tqdm import tqdm
from datetime import datetime
from .Optimizers import MPGA, PSO, SGA, SA
from GQLib.Optimizers import Optimizer
from GQLib.LombAnalysis import LombAnalysis
from GQLib.Models import LPPL, LPPLS
import plotly.graph_objects as go


class Framework:
    """
    Framework for processing and analyzing financial time series using LPPL and Lomb-Scargle techniques.

    This framework includes:
    - Data loading and subinterval generation.
    - Optimization of LPPL parameters using a custom optimizer.
    - Lomb-Scargle periodogram analysis for detecting significant frequencies.
    - Visualization of results, including LPPL predictions and significant critical times.
    """

    def __init__(self, frequency: str = "daily", is_uso : bool = False) -> None:
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

        self.data = self.load_data(is_uso)

        self.global_times = self.data[:, 0].astype(float)
        self.global_dates = self.data[:, 1]
        self.global_prices = self.data[:, 2].astype(float)

    def load_data(self, is_uso) -> np.ndarray:
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
        if is_uso == False:
            data = pd.read_csv(f'data/WTI_Spot_Price_{self.frequency}.csv', skiprows=4)
            data.columns = ["Date", "Price"]
            data["Date"] = pd.to_datetime(data["Date"], format="%m/%d/%Y").values.astype("datetime64[D]")
        else:
            data = pd.read_csv(f'data/USO_{self.frequency}.csv', sep=";")
            data['Price'] = data['Price'].apply(lambda x:x/8) # Stock split 1:8 en 2020
            data["Date"] = pd.to_datetime(data["Date"], format="%d/%m/%Y").values.astype("datetime64[D]")


        # Date conversion and sorting
        
        data = data.sort_values(by="Date")

        # Add numeric time index
        t = np.linspace(0, len(data) - 1, len(data))
        data = np.insert(data.to_numpy(), 0, t, axis=1)

        return data


    def process(self, time_start: str, time_end: str, optimizer: Optimizer) -> None:
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
                use_package: bool = False,
                remove_mpf: bool = True,
                mpf_threshold: float = 1e-3,
                show: bool = False,
                lppl_model: 'LPPL | LPPLS' = LPPL) -> dict:
        """
        Analyze results using Lomb-Scargle periodogram and identify significant critical times.

        Parameters
        ----------
        results : dict
            Optimization results to analyze.
        result_json_name : dict, optional
            Path to a JSON file containing results. If None, uses `self.results`.
        remove_mpf : bool, optional
            Whether to remove the "most probable frequency" from the results. Default is True.
        mpf_threshold : float, optional
            Threshold for filtering frequencies close to the most probable frequency. Default is 1e-3.
        show : bool, optional
            Whether to display visualizations of the Lomb spectrum and LPPL fits. Default is False.
        lppl_model : (LPPL | LPPLS)
            An instance of the LPPL or LPPLS model with fitted parameters.
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

        for idx, res in enumerate(tqdm(results, desc="Analyzing results", unit="result")):

            mask = (self.global_times >= res["sub_start"]) & (self.global_times <= res["sub_end"])
            t_sub = self.global_times[mask]
            y_sub = self.global_prices[mask]

            # Lomb-Scargle analysis
            lomb = LombAnalysis(lppl_model(t_sub, y_sub, res["bestParams"]))
            lomb.compute_lomb_periodogram(use_package=use_package)
            lomb.filter_results(remove_mpf=remove_mpf, mpf_threshold=mpf_threshold)
            is_significant = lomb.check_significance()

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



            best_results.append({
                "sub_start": res["sub_start"],
                "sub_end": res["sub_end"],
                "bestObjV": res["bestObjV"],
                "bestParams": res["bestParams"],
                "is_significant": is_significant
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

    def visualize(self, best_results : dict, name = "", start_date: str = None, end_date: str = None) -> None:
        """
        Visualize significant critical times on the price series.
        Permet de filtrer et d'afficher les résultats pour une plage de dates spécifique.
        
        Args:
            best_results (dict): Résultats optimaux contenant les informations des turning points.
            name (str): Nom du graphique.
            start_date (str): Date de début (format: 'YYYY-MM-DD'). Si None, utilise le début des données.
            end_date (str): Date de fin (format: 'YYYY-MM-DD'). Si None, utilise la fin des données.
        """
        significant_tc = []
        min_time = np.inf
        max_time = -np.inf

        for res in best_results:
            if res["sub_start"] < min_time:
                min_time = res["sub_start"]
            if res["sub_end"] > max_time:
                max_time = res["sub_end"]
            if res["is_significant"]:
                significant_tc.append(res["bestParams"][0])

        if start_date is not None and end_date is not None:
            start_date = pd.to_datetime(start_date, format="%d/%m/%Y") - timedelta(days=5*365)
            end_date = pd.to_datetime(end_date, format="%d/%m/%Y") + timedelta(days=5*365)
        else:
            start_date = self.global_dates.min()
            end_date = self.global_dates.max()

        filtered_indices = [i for i, date in enumerate(self.global_dates) if start_date <= date <= end_date]
        if not filtered_indices:
            print(f"Aucune donnée disponible entre {start_date} et {end_date}.")
            return

        filtered_dates = [self.global_dates[i] for i in filtered_indices]
        filtered_prices = [self.global_prices[i] for i in filtered_indices]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=filtered_dates,
                y=filtered_prices,
                mode='lines',
                name='Data'
                # line=dict(color='black')
            )
        )
        if start_date <= self.global_dates[int(min_time)] <= end_date:
            fig.add_trace(
                go.Scatter(
                    x=[self.global_dates[int(min_time)], self.global_dates[int(min_time)]],
                    y=[min(filtered_prices) - 10, max(filtered_prices) + 10],
                    mode="lines",
                    line=dict(color="gray", dash="dash"),
                    name="Start Date",
                    showlegend=True
                )
            )

        if start_date <= self.global_dates[int(max_time)] <= end_date:
            fig.add_trace(
                go.Scatter(
                    x=[self.global_dates[int(max_time)], self.global_dates[int(max_time)]],
                    y=[min(filtered_prices) - 10, max(filtered_prices) + 10],
                    mode="lines",
                    line=dict(color="gray", dash="longdash"),
                    name="End Date",
                    showlegend=True
                )
            )

        # Ajout des temps critiques (Critical Time)
        for idx, tc in enumerate(significant_tc):
            try:
                date_tc = self.global_dates[int(round(tc))]
                if start_date <= date_tc <= end_date:
                    # Lignes verticales pour les temps critiques
                    fig.add_trace(
                        go.Scatter(
                            x=[date_tc, date_tc],
                            y=[min(filtered_prices)-10, max(filtered_prices)+10],
                            mode="lines",
                            line=dict(color="red", dash="dot"),
                            name="Critical Time" if idx == 0 else None,  # Légende uniquement pour le premier
                            showlegend=(idx == 0)
                        )
                    )
            except:
                continue
        fig.update_layout(
            title=name,
            xaxis_title="Date",
            yaxis_title="Price",
            showlegend=True,
        )
        fig.show()

    def compare_results(self, multiple_results: dict[str, dict], name:str = "", data_name: str = "",
                        real_tc: str = None, start_date: str = None, end_date: str = None):
        """
        Visualize multiple run results

        Parameters
        ----------
        multiple_results (list[dict]) : list of each run results
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
        if start_date is not None and end_date is not None:
            start_date = pd.to_datetime(start_date, format="%d/%m/%Y") - timedelta(days=1*365)
            end_date = pd.to_datetime(end_date, format="%d/%m/%Y") + timedelta(days=5*365)
        else:
            start_date = self.global_dates.min()
            end_date = self.global_dates.max()

        filtered_indices = [i for i, date in enumerate(self.global_dates) if start_date <= date <= end_date]
        if not filtered_indices:
            print(f"Aucune donnée disponible entre {start_date} et {end_date}.")
            return

        filtered_dates = [self.global_dates[i] for i in filtered_indices]
        filtered_prices = [self.global_prices[i] for i in filtered_indices]

        fig = go.Figure()
        # Plot de la série de prix
        fig.add_trace(
            go.Scatter(
                x=filtered_dates,
                y=filtered_prices,
                mode='lines',
                name=data_name
            )
        )
        # Si la vraie date du tc est fournie, on la plot
        if real_tc is not None:
            target_date = pd.to_datetime(real_tc, format="%d/%m/%Y")

            for entry in self.data:
                if entry[1] == target_date:
                    time_tc = entry[0]

            if time_tc:
                fig.add_trace(
                    go.Scatter(
                        x=[self.global_dates[int(time_tc)], self.global_dates[int(time_tc)]],
                        y=[min(filtered_prices) - 10, max(filtered_prices) + 10],
                        mode="lines",
                        line=dict(color="red", width=4),
                        textfont=dict(weight='bold'),
                        name="Real critical time",
                        showlegend=True,
                        opacity=1
                    )
                )

        # Pour chaque modèle, on plot les tc
        for i, model in enumerate(multiple_results.keys()):
            best_results = multiple_results[model]
            significant_tc = []
            min_time = np.inf
            max_time = -np.inf

            for res in best_results:
                if res["sub_start"] < min_time:
                    min_time = res["sub_start"]
                if res["sub_end"] > max_time:
                    max_time = res["sub_end"]
                if res["is_significant"]:
                    significant_tc.append(res["bestParams"][0])

            # On plot les start et end date une fois à la première itération
            if i == 0:
                if start_date <= self.global_dates[int(min_time)] <= end_date:
                    fig.add_trace(
                        go.Scatter(
                            x=[self.global_dates[int(min_time)], self.global_dates[int(min_time)]],
                            y=[min(filtered_prices) - 10, max(filtered_prices) + 10],
                            mode="lines",
                            line=dict(color="gray", dash="dash"),
                            name="Start Date",
                            showlegend=True
                        )
                    )

                if start_date <= self.global_dates[int(max_time)] <= end_date:
                    fig.add_trace(
                        go.Scatter(
                            x=[self.global_dates[int(max_time)], self.global_dates[int(max_time)]],
                            y=[min(filtered_prices) - 10, max(filtered_prices) + 10],
                            mode="lines",
                            line=dict(color="gray", dash="longdash"),
                            name="End Date",
                            showlegend=True
                        )
                    )

            index_plot = 0
            for tc in significant_tc:
                try:
                    date_tc = self.global_dates[int(round(tc))]
                    if start_date <= date_tc <= end_date:
                        fig.add_trace(
                            go.Scatter(
                                x=[date_tc, date_tc],
                                y=[min(filtered_prices) - 10, max(filtered_prices) + 10],
                                mode="lines",
                                line=dict(color=colors[i], dash="dot"),
                                name=f"{model} critical times" if index_plot == 0 else None,
                                showlegend=(index_plot == 0)
                            )
                        )
                        index_plot += 1
                except:
                    continue

        fig.update_layout(
            title=name,
            xaxis_title="Date",
            yaxis_title="Price",
            showlegend=True,
        )
        fig.show()

    def compare_results_rectangle(self, multiple_results: dict[str, dict], name: str = "", data_name: str = "",
                                  real_tc: str = None, start_date: str = None, end_date: str = None):
        """
        Visualize multiple run results

        Parameters
        ----------
        multiple_results (dict[str, dict]): Dict of each model's results
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

        if start_date is not None and end_date is not None:
            start_date = pd.to_datetime(start_date, format="%d/%m/%Y") - timedelta(days=1 * 365)
            end_date = pd.to_datetime(end_date, format="%d/%m/%Y") + timedelta(days=5 * 365)
        else:
            start_date = self.global_dates.min()
            end_date = self.global_dates.max()

        filtered_indices = [i for i, date in enumerate(self.global_dates) if start_date <= date <= end_date]
        if not filtered_indices:
            print(f"Aucune donnée disponible entre {start_date} et {end_date}.")
            return

        filtered_dates = [self.global_dates[i] for i in filtered_indices]
        filtered_prices = [self.global_prices[i] for i in filtered_indices]

        fig = go.Figure()

        # Plot de la série de prix
        fig.add_trace(
            go.Scatter(
                x=filtered_dates,
                y=filtered_prices,
                mode='lines',
                name=data_name
            )
        )

        # Si la vraie date du tc est fournie, on la plot
        if real_tc is not None:
            target_date = pd.to_datetime(real_tc, format="%d/%m/%Y")

            for entry in self.data:
                if entry[1] == target_date:
                    time_tc = entry[0]

            if time_tc:
                fig.add_trace(
                    go.Scatter(
                        x=[self.global_dates[int(time_tc)], self.global_dates[int(time_tc)]],
                        y=[min(filtered_prices)-10, max(filtered_prices)+10],
                        mode="lines",
                        line=dict(color="red", width=4),
                        name="Real critical time",
                        showlegend=True
                    )
                )

        # Je veux garder 1/5 du max de la time series en haut et en bas
        total_height = int(max(filtered_prices))
        base_y = total_height / 5
        remaining_height = total_height - 2 * base_y

        # On divise l'espace en restant pour que chaque model ait la même hauteur
        rectangle_height = remaining_height / len(multiple_results.keys())

        for i, model in enumerate(multiple_results.keys()):
            best_results = multiple_results[model]
            significant_tc = []
            min_time = np.inf
            max_time = -np.inf

            for res in best_results:
                if res["sub_start"] < min_time:
                    min_time = res["sub_start"]
                if res["sub_end"] > max_time:
                    max_time = res["sub_end"]
                if res["is_significant"]:
                    significant_tc.append(res["bestParams"][0])

            # On plot les start et end date une fois à la première itération
            if i == 0:
                if start_date <= self.global_dates[int(min_time)] <= end_date:
                    fig.add_trace(
                        go.Scatter(
                            x=[self.global_dates[int(min_time)], self.global_dates[int(min_time)]],
                            y=[min(filtered_prices) - 10, max(filtered_prices) + 10],
                            mode="lines",
                            line=dict(color="gray", dash="dash"),
                            name="Start Date",
                            showlegend=True
                        )
                    )

                if start_date <= self.global_dates[int(max_time)] <= end_date:
                    fig.add_trace(
                        go.Scatter(
                            x=[self.global_dates[int(max_time)], self.global_dates[int(max_time)]],
                            y=[min(filtered_prices) - 10, max(filtered_prices) + 10],
                            mode="lines",
                            line=dict(color="gray", dash="longdash"),
                            name="End Date",
                            showlegend=True
                        )
                    )

            if significant_tc:
                min_tc_date = self.global_dates[int(round(min(significant_tc)))]
                max_tc_date = self.global_dates[int(round(max(significant_tc)))]

                # Rectangle pour le modèle
                fig.add_trace(
                    go.Scatter(
                        x=[min_tc_date, max_tc_date, max_tc_date, min_tc_date, min_tc_date],
                        y=[base_y + i * rectangle_height, base_y + i * rectangle_height,
                           base_y + (i + 1) * rectangle_height,
                           base_y + (i + 1) * rectangle_height, base_y + i * rectangle_height],
                        fill="toself",
                        fillcolor=colors[i % len(colors)],
                        line=dict(color=colors[i % len(colors)], width=1),
                        name=f"{model} critical time",
                        opacity=0.5,
                        showlegend=False
                    )
                )

                # Ajout du nom du modèle au centre du rectangle
                center_x = min_tc_date + (max_tc_date - min_tc_date) / 2
                center_y = base_y + i * rectangle_height + rectangle_height / 2
                fig.add_trace(
                    go.Scatter(
                        x=[center_x],
                        y=[center_y],
                        text=[model],
                        mode="text",
                        showlegend=False
                    )
                )

        fig.update_layout(
            title=name,
            xaxis_title="Date",
            yaxis_title="Price",
            showlegend=True
        )
        fig.show()


    def generate_all_dates(self, optimizers : list =  [PSO(), MPGA(), SA(), SGA()]):
        dates_sets = {
            "Set 1": ("01/04/2003", "02/01/2008"),
            "Set 2": ("01/02/2007", "01/02/2011"),
            "Set 3": ("29/04/2011", "01/08/2015"),
        }
        print(f"FREQUENCY : {self.frequency}")
        for optimizer in optimizers:
            print(f"Running process for {optimizer.__class__.__name__}\n")
            for set_name, (start_date, end_date) in dates_sets.items():
                print(f"Running process for {set_name} from {start_date} to {end_date}")
                # Exécute le processus d'optimisation pour l'intervalle de dates donné
                results = self.process(start_date, end_date, optimizer)
                # Conversion des chaînes de dates en objets datetime pour faciliter le formatage
                start_date_obj = datetime.strptime(start_date, "%d/%m/%Y")
                end_date_obj = datetime.strptime(end_date, "%d/%m/%Y")
                filename = f"results/{optimizer.__class__.__name__}/{self.frequency}/{start_date_obj.strftime('%m-%Y')} {end_date_obj.strftime('%m-%Y')}.json"
                # Sauvegarde des résultats au format JSON dans le fichier généré
                self.save_results(results, filename)
                # Verification de la significativité des résultats
                best_results = self.analyze(results)
                # Visualisation des résultats finaux
                self.visualize(best_results, optimizer.__class__.__name__)

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
        with open(file_name, "w") as f:
            json.dump(results, f, indent=4)