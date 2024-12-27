from GQLib.Framework import Framework
from GQLib.Optimizers import MPGA, SA

fw = Framework("daily")

# On renseigne les dates du sample et l'optimiseur à utiliser
fw.process("04/01/2003", "04/02/2008", SA)

# On peut save en json nos résultats pour les réutiliser plus tard
fw.save_results(fw.results, "results-2003_2008_try_PSO.json")

# On check la significativité des résultats
fw.analyze()

# On peut visualiser les résultats finaux
fw.visualize()

print("done")