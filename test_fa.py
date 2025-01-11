from GQLib.Framework import Framework
from GQLib.Optimizers import MPGA, PSO, SA, SGA, FA
from GQLib.Models import LPPL, LPPLS
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

fw = Framework("daily", lppl_model = LPPL)
# On renseigne les dates du sample et l'optimiseur à utiliser
optimizer = FA()
results = fw.process("04/01/2003", "04/02/2008", optimizer= optimizer)
# On peut save en json nos résultats pour les réutiliser plus tard
#fw.save_results(results, "results/results-2003_2008_try_PSO.json")
# On check la significativité des résultats
best_results = fw.analyze(results)
# On peut visualiser les résultats finaux
fw.visualize(best_results)
