from datetime import datetime
from .Optimizers import MPGA, PSO, SGA, SA
from .Framework import Framework
from GQLib.Optimizers import Optimizer
from GQLib.LombAnalysis import LombAnalysis
from GQLib.Models import LPPL, LPPLS
import plotly.graph_objects as go

def generate_all_dates( frequency : str = "daily",
                        optimizers : list[Optimizer] =  [SA(), SGA(), PSO(), MPGA()], 
                        nb_tc : int = None, 
                        significativity_tc = 0.3,
                        save : bool = False):
    
    if frequency not in ["daily", "weekly", "monthly"]:
            raise ValueError("The frequency must be one of 'daily', 'weekly', 'monthly'.")
    
    #Initialisation des dates
    dates_sets = {
        "Set 1": ("01/04/2003", "02/01/2008"),
        "Set 2": ("01/02/2007", "01/02/2011"),
        "Set 3": ("29/04/2011", "01/08/2015"),
    }
    dates_graphs = [
    ("01/10/2003", "31/12/2009"),
    ("01/12/2008", "31/12/2012"),
    ("01/11/2011", "31/12/2016"),
    ]

    #Initialisation du Framework
    fw = Framework(frequency = frequency)
    print(f"FREQUENCY : {frequency}")

    for optimizer in optimizers:
        fw.lppl_model = optimizer.lppl_model
        current = 0
        print(f"\nRunning process for {optimizer.__class__.__name__}")
        for set_name, (start_date, end_date) in dates_sets.items():

            graph_start_date, graph_end_date = dates_graphs[current]
            print(f"Running process for {set_name} from {start_date} to {end_date}")

            # Exécute le processus d'optimisation pour l'intervalle de dates donné
            results = fw.process(start_date, end_date, optimizer)

            # Conversion des chaînes de dates en objets datetime pour faciliter le formatage
            start_date_obj = datetime.strptime(start_date, "%d/%m/%Y")
            end_date_obj = datetime.strptime(end_date, "%d/%m/%Y")
            filename = f"results/{optimizer.__class__.__name__}/{frequency}/{optimizer.lppl_model.__name__}_{start_date_obj.strftime('%m-%Y')}_{end_date_obj.strftime('%m-%Y')}.json"
            
            if save:
                # Sauvegarde des résultats au format JSON dans le fichier généré
                fw.save_results(results, filename)

            # Verification de la significativité des résultats
            best_results = fw.analyze(results, significativity_tc=significativity_tc)
            # Visualisation des résultats finaux
            fw.visualize(
                best_results,
                optimizer.__class__.__name__,
                start_date=graph_start_date,
                end_date=graph_end_date,
                nb_tc = nb_tc
            )
            current+=1

def generate_all_rectangle(frequency : str = "daily",
                            optimizers: list[Optimizer] = [SA(), SGA(), PSO(), MPGA()],
                            significativity_tc=0.3,
                            nb_tc : int = 20,
                            rerun: bool = False,
                            save: bool = False,
                            save_plot : bool = False):
    dates_sets = {
        "Set 1": ("01/04/2003", "02/01/2008"),
        "Set 2": ("01/02/2007", "01/02/2011"),
        "Set 3": ("29/04/2011", "01/08/2015"),
    }
    if frequency not in ["daily", "weekly", "monthly"]:
            raise ValueError("The frequency must be one of 'daily', 'weekly', 'monthly'.")
    fw = Framework(frequency = frequency)
    real_tcs = ['03/07/2008', '29/04/2011', '11/02/2016']
    optimiseurs_models = []
    print(f"FREQUENCY : {frequency}")
    compteur = 0

    for set_name, (start_date, end_date) in dates_sets.items():
        print(f"Running process for {set_name} from {start_date} to {end_date}")
        best_results_list = {}

        for optimizer in optimizers:
            fw.lppl_model = optimizer.lppl_model
            optimiseurs_models.append(fw.lppl_model.__name__)
            # Conversion des chaînes de dates en objets datetime pour faciliter le formatage
            start_date_obj = datetime.strptime(start_date, "%d/%m/%Y")
            end_date_obj = datetime.strptime(end_date, "%d/%m/%Y")
            filename = f"results/{optimizer.__class__.__name__}/{frequency}/{ optimizer.lppl_model.__name__}_{start_date_obj.strftime('%m-%Y')}_{end_date_obj.strftime('%m-%Y')}.json"
            
            if rerun:
                print(f"\nRunning process for {optimizer.__class__.__name__}")
                results = fw.process(start_date, end_date, optimizer)
                best_results_list[optimizer.__class__.__name__] = fw.analyze(results=results,
                                                                                significativity_tc=significativity_tc)
                if save:
                    fw.save_results(results, filename)
            else:
                print(f"Getting result for {optimizer.__class__.__name__}\n")
                best_results_list[optimizer.__class__.__name__] = fw.analyze(result_json_name=filename,
                                                                                significativity_tc=significativity_tc)

        fw.compare_results_rectangle(multiple_results=best_results_list, 
                                    name=f"Predicted critical times {frequency}",
                                    data_name="WTI Data", 
                                    real_tc=real_tcs[compteur], 
                                    optimiseurs_models = optimiseurs_models,
                                    start_date=start_date,
                                    end_date=end_date,
                                    nb_tc = nb_tc,
                                    save_plot = save_plot)
        compteur += 1
