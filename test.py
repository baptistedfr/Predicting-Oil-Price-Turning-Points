from GQLib.Framework import Framework
from GQLib.Optimizers import MPGA, PSO, SGA, SA, NELDER_MEAD, TABU, FA
from GQLib.Models import LPPL, LPPLS
from GQLib.enums import InputType
from GQLib.AssetProcessor import AssetProcessor
import numpy as np


wti = AssetProcessor(input_type = InputType.WTI)
# wti.generate_all_dates(frequency= "daily",
#                         optimizers =  [SA(LPPL), SGA(LPPL), NELDER_MEAD(LPPLS), TABU(LPPL), FA(LPPL), MPGA(LPPL), PSO(LPPL)], 
#                         rerun = False,
#                         nb_tc = None, 
#                         significativity_tc = 0.3,
#                         save  = False)

significativity = np.arange(0.05,0.35, 0.05)
tcs = np.arange(5,25, 5)

for sig in significativity:
    for tc in tcs:
        wti.generate_all_rectangle(frequency = "daily",
                            optimizers =  [SA(LPPL), SGA(LPPL), NELDER_MEAD(LPPLS), TABU(LPPL), FA(LPPL), MPGA(LPPL), PSO(LPPL)], 
                            significativity_tc=sig,
                            rerun = False,
                            nb_tc = tc,
                            save=False,
                            save_plot=False)

# wti.generate_all_rectangle(frequency = "daily",
#                         optimizers =  [SA(LPPL), SGA(LPPL), NELDER_MEAD(LPPLS), TABU(LPPL), FA(LPPL), MPGA(LPPL), PSO(LPPL)], 
#                         significativity_tc=0.15,
#                         rerun = False,
#                         nb_tc = 10,
#                         save=False,
#                         save_plot=False)

# wti.generate_all_rectangle(frequency = "daily",
#                         optimizers =  [SA(LPPL), SGA(LPPL), NELDER_MEAD(LPPLS), TABU(LPPL), FA(LPPL), MPGA(LPPL), PSO(LPPL)], 
#                         significativity_tc=0.15,
#                         rerun = False,
#                         nb_tc = 15,
#                         save=False,
#                         save_plot=False)


# sp500 = AssetProcessor(input_type = InputType.SP500)
# sp500.generate_all_rectangle(frequency = "daily",
#                         optimizers =  [SA(LPPL), SGA(LPPL), NELDER_MEAD(LPPLS)], 
#                         significativity_tc=0.3,
#                         rerun = True,
#                         nb_tc = 20,
#                         save=True,
#                         save_plot=True)

