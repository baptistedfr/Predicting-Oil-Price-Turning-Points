from GQLib.Framework import Framework
from GQLib.Optimizers import MPGA, PSO, SA, SGA
from datetime import datetime
fw = Framework("daily")
best_results = fw.analyze(result_json_name="results/MPGA/daily/02-2007 02-2011.json")
# On peut visualiser les résultats finaux
fw.visualize(best_results, start_date="02/02/2011", end_date="01/01/2012")