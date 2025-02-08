from GQLib.Framework import Framework
from GQLib.Optimizers import MPGA, PSO, SGA, SA, NELDER_MEAD, TABU, FA
from GQLib.Models import LPPL, LPPLS
from GQLib.enums import InputType
from GQLib.AssetProcessor import AssetProcessor
import numpy as np




wti = AssetProcessor(input_type = InputType.BTC)


# wti.compare_optimizers(frequency = "daily",
#                             optimizers =  [SA(LPPLS), PSO(LPPLS), MPGA(LPPLS),SGA(LPPLS), TABU(LPPLS), FA(LPPLS), NELDER_MEAD(LPPLS)],
#                             significativity_tc=0.3,
#                             rerun = False,
#                             nb_tc = 10,
#                             save=False,
#                             save_plot=False)

# wti.compare_optimizers(frequency = "daily",
#                             optimizers =  [SA(LPPLS), PSO(LPPLS), MPGA(LPPLS),SGA(LPPLS), TABU(LPPLS), FA(LPPLS), NELDER_MEAD(LPPLS)],
#                             significativity_tc=0.3,
#                             rerun = False,
#                             nb_tc = 10,
#                             save=False,
#                             save_plot=False)

# wti.compare_optimizers(frequency = "daily",
#                             optimizers =  [SA(LPPL), PSO(LPPL), MPGA(LPPL),SGA(LPPL), TABU(LPPL), FA(LPPL), NELDER_MEAD(LPPLS)],
#                             significativity_tc=0.3,
#                             rerun = False,
#                             nb_tc = 10,
#                             save=False,
#                             save_plot=False)

# wti.compare_optimizers(frequency = "weekly",
#                             optimizers =  [SA(LPPLS), PSO(LPPLS), MPGA(LPPLS),SGA(LPPLS), TABU(LPPLS), FA(LPPLS), NELDER_MEAD(LPPLS)],
#                             significativity_tc=0.3,
#                             rerun = False,
#                             nb_tc = 10,
#                             save=False,
#                             save_plot=False)

# wti.compare_optimizers(frequency = "weekly",
#                             optimizers =  [SA(LPPLS), PSO(LPPLS), MPGA(LPPLS),SGA(LPPLS), TABU(LPPLS), FA(LPPLS), NELDER_MEAD(LPPLS)],
#                             significativity_tc=0.3,
#                             rerun = False,
#                             nb_tc = 10,
#                             save=False,
#                             save_plot=False)

# wti.visualise_tc(frequency = "daily",
#                             optimizers =  [SA(LPPL), SGA(LPPL), NELDER_MEAD(LPPLS), TABU(LPPL), FA(LPPL), MPGA(LPPL), PSO(LPPL)],
#                             significativity_tc=0.3,
#                             rerun = False,
#                             nb_tc = 10,
#                             save=False)
