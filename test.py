from GQLib.Framework import Framework
from GQLib.Optimizers import MPGA, PSO, SA, SGA
from datetime import datetime

fw = Framework("daily")
# optimizer = MPGA()
# # On renseigne les dates du sample et l'optimiseur à utiliser
# results = fw.process("04/01/2003", "04/02/2008", optimizer)
# # On visualise la convergence de l'algorithm
# optimizer.visualize_convergence()
# # On peut save en json nos résultats pour les réutiliser plus tard
# fw.save_results(results, "results/results-2003_2008_try_PSO.json")
# # On check la significativité des résultats
# fw.analyze(results, "results/results-2003_2008_try_PSO.json")
# # On peut visualiser les résultats finaux
# fw.visualize(results)
fw.generate_all_dates()