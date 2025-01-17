from datetime import datetime
import json
from .Optimizers import MPGA, PSO, SGA, SA
from .Framework import Framework
from GQLib.Optimizers import Optimizer
from .enums import InputType


class AssetProcessor:
    """
    This class processes financial asset data, applies optimization algorithms on LPPL Models,
    and visualizes the results based on various configurations.
    """
    def __init__(self, input_type : InputType = InputType.WTI):
        """
        Initializes the AssetProcessor with a specified input type (e.g., WTI).
        
        Args:
            input_type (InputType): The type of asset data to process (default is WTI).
        """
        self.input_type = input_type
        print(self.input_type.value)
        # On load la config de notre input_type 
        config = self.load_config()
        self.dates_sets = config["sets"]
        self.dates_graphs = config["graphs"]
        self.real_tcs = config["real_tcs"]

    def load_config(self):
        """
        Loads the configuration for the specified input type from a JSON file.
        
        Returns:
            dict: A dictionary containing the configuration data for the specified input type.
        
        Raises:
            ValueError: If the input type is not found in the configuration file.
        """
        with open("params/config_asset_classes.json", "r") as file:
            config = json.load(file)
        if self.input_type.name not in config:
            raise ValueError(f"Input type {self.input_type.name} not found in configuration.")
        return config[self.input_type.name]

    def visualise_tc(self,
                           frequency : str = "daily",
                           optimizers : list[Optimizer] =  [SA(), SGA(), PSO(), MPGA()], 
                           rerun : bool = False,
                           nb_tc : int = None, 
                           significativity_tc = 0.3,
                           save : bool = False):
        
        """
        Visualizes the turning points (TC) for a given frequency and optimization method.
        The list of dates for each data is defined in the config file

        Args:
            frequency (str): The frequency of the data ('daily', 'weekly', or 'monthly').
            optimizers (list[Optimizer]): A list of optimizers to use for the analysis.
            rerun (bool): Whether to rerun the optimization process (default is False).
            nb_tc (int): The number of turning points to visualize (default is None, meaning all).
            significativity_tc (float): The significance threshold for the turning points (default is 0.3).
            save (bool): Whether to save the results as JSON files (default is False).
        """
    
        if frequency not in ["daily", "weekly", "monthly"]:
                raise ValueError("The frequency must be one of 'daily', 'weekly', 'monthly'.")
        
        #Initialisation du Framework
        fw = Framework(frequency = frequency, input_type=self.input_type)
        print(f"FREQUENCY : {frequency}")

        # Visualisation pour chaque algorithme
        for optimizer in optimizers:
            current = 0
            print(f"\nRunning process for {optimizer.__class__.__name__}")
            for set_name, (start_date, end_date) in self.dates_sets.items():

                graph_start_date, graph_end_date = self.dates_graphs[current]
                # Conversion des chaînes de dates en objets datetime pour faciliter le formatage
                start_date_obj = datetime.strptime(start_date, "%d/%m/%Y")
                end_date_obj = datetime.strptime(end_date, "%d/%m/%Y")
                filename = f"Results/results_{self.input_type.value}/{optimizer.__class__.__name__}/{frequency}/{optimizer.lppl_model.__name__}_{start_date_obj.strftime('%m-%Y')}_{end_date_obj.strftime('%m-%Y')}.json"
                
                if rerun : 
                    print(f"Running process for {set_name} from {start_date} to {end_date}")

                    # Exécute le processus d'optimisation pour l'intervalle de dates donné
                    results = fw.process(start_date, end_date, optimizer)
                    if save:
                        # Sauvegarde des résultats au format JSON dans le fichier généré
                        fw.save_results(results, filename)
                    # Verification de la significativité des résultats
                    best_results = fw.analyze(results, significativity_tc=significativity_tc, lppl_model = optimizer.lppl_model)

                else:
                    best_results = fw.analyze(result_json_name=filename,significativity_tc=significativity_tc,lppl_model=optimizer.lppl_model)
                
                # Visualisation des résultats finaux
                fw.visualize_tc(
                    best_results,
                    f"{self.input_type.value} {optimizer.__class__.__name__} {frequency} ({optimizer.lppl_model.__name__}) results from {start_date_obj.strftime('%m-%Y')} to {end_date_obj.strftime('%m-%Y')}",
                    start_date=graph_start_date,
                    end_date=graph_end_date,
                    nb_tc = nb_tc,
                    real_tc = self.real_tcs[current]
                )
                current+=1

    def compare_optimizers(self,
                               frequency : str = "daily",
                               optimizers: list[Optimizer] = [SA(), SGA(), PSO(), MPGA()],
                               significativity_tc=0.3,
                               nb_tc : int = 20,
                               rerun: bool = False,
                               save: bool = False,
                               save_plot : bool = False):
    
        """
        Compares the performance of different optimizers in predicting turning points.

        Args:
            frequency (str): The frequency of the data ('daily', 'weekly', or 'monthly').
            optimizers (list[Optimizer]): A list of optimizers to compare.
            significativity_tc (float): The significance threshold for the turning points (default is 0.3).
            nb_tc (int): The number of turning points to visualize (default is 20).
            rerun (bool): Whether to rerun the optimization process (default is False).
            save (bool): Whether to save the results as JSON files (default is False).
            save_plot (bool): Whether to save the comparison plot (default is False).
        """
        if frequency not in ["daily", "weekly", "monthly"]:
                raise ValueError("The frequency must be one of 'daily', 'weekly', 'monthly'.")
        
        fw = Framework(frequency = frequency, input_type=self.input_type)
        
        print(f"FREQUENCY : {frequency}")
        compteur = 0

        for set_name, (start_date, end_date) in self.dates_sets.items():
            print(f"Running process for {set_name} from {start_date} to {end_date}")
            best_results_list = {}
            optimiseurs_models = []

            for optimizer in optimizers:
                optimiseurs_models.append(optimizer.lppl_model.__name__)
                # Conversion des chaînes de dates en objets datetime pour faciliter le formatage
                start_date_obj = datetime.strptime(start_date, "%d/%m/%Y")
                end_date_obj = datetime.strptime(end_date, "%d/%m/%Y")
                filename = f"Results/results_{self.input_type.value}/{optimizer.__class__.__name__}/{frequency}/{ optimizer.lppl_model.__name__}_{start_date_obj.strftime('%m-%Y')}_{end_date_obj.strftime('%m-%Y')}.json"
                
                if rerun:
                    print(f"\nRunning process for {optimizer.__class__.__name__}")
                    results = fw.process(start_date, end_date, optimizer)
                    best_results_list[optimizer.__class__.__name__] = fw.analyze(results=results,
                                                                                    significativity_tc=significativity_tc,
                                                                                    lppl_model=optimizer.lppl_model)
                    optimizer.visualize_convergence()
                    if save:
                        fw.save_results(results, filename)
                else:
                    print(f"Getting result for {optimizer.__class__.__name__}\n")
                    best_results_list[optimizer.__class__.__name__] = fw.analyze(result_json_name=filename,
                                                                                    significativity_tc=significativity_tc,
                                                                                    lppl_model=optimizer.lppl_model)


            real_tc = self.real_tcs[compteur] if len(self.real_tcs)>compteur else None
            fw.visualize_compare_results(multiple_results=best_results_list, 
                                        name=f"Predicted critical times {frequency} {self.input_type.value} from {start_date_obj.strftime('%m-%Y')} to {end_date_obj.strftime('%m-%Y')}",
                                        data_name=f"{self.input_type.value} Data", 
                                        real_tc=real_tc, 
                                        optimiseurs_models = optimiseurs_models,
                                        start_date=start_date,
                                        end_date=end_date,
                                        nb_tc = nb_tc,
                                        save_plot = save_plot)
            compteur += 1

    def visualise_data(self,
                       frequency : str = "daily",
                       start_date = None,
                        end_date = None):
        """
        Visualizes the raw data for a given frequency and date range.
        
        Args:
            frequency (str): The frequency of the data ('daily', 'weekly', or 'monthly').
            start_date (str): The start date for the visualization (default is None).
            end_date (str): The end date for the visualization (default is None).
        """

        if frequency not in ["daily", "weekly", "monthly"]:
                raise ValueError("The frequency must be one of 'daily', 'weekly', 'monthly'.")
        
        fw = Framework(frequency = frequency, input_type=self.input_type)
        fw.visualise_data(start_date, end_date)