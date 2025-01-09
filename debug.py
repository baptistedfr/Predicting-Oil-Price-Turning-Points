from GQLib.Framework import Framework
from GQLib.Optimizers import MPGA, PSO, SA, SGA, MCMC, NELDER_MEAD
from datetime import datetime
from GQLib.Models import LPPL, LPPLS

freq = "daily"
model = LPPL

fw = Framework(freq, model)
optimizers = [MCMC]
dates_sets = {
    "Set 1": ("01/04/2003", "02/01/2008"),
    "Set 2": ("01/02/2007", "01/02/2011"),
    "Set 3": ("29/04/2011", "01/08/2015"),
}

for optimizer in optimizers:
    print(f"Running process for {optimizer.__name__}\n")
    for set_name, (start_date, end_date) in dates_sets.items():
        print(f"Running process for {set_name} from {start_date} to {end_date}")
        fw.process(start_date, end_date, optimizer)
        # On peut save en json nos résultats pour les réutiliser plus tard
        start_date_obj = datetime.strptime(start_date, "%d/%m/%Y")
        end_date_obj = datetime.strptime(end_date, "%d/%m/%Y")
        filename = f"results/{optimizer.__name__}/{model.__name__}-results_{freq}_{start_date_obj.strftime('%d-%m-%y')} {end_date_obj.strftime('%d-%m-%y')}_{optimizer.__name__}.json"
        # Sauvegarder les résultats
        fw.save_results(fw.results, filename)
        # On check la significativité des résultats
        fw.analyze()
        # On peut visualiser les résultats finaux
        fw.visualize()