from GQLib.Framework import Framework
from GQLib.Optimizers import MPGA, PSO, SGA, SA, NELDER_MEAD, TABU, FA, MCMC
from GQLib.Models import LPPL, LPPLS
from GQLib.enums import InputType
from GQLib.AssetProcessor import AssetProcessor
import numpy as np

wti = AssetProcessor(input_type = InputType.WTI)
mcmc = MCMC(LPPL)
wti.compare_optimizers(frequency = "daily",
                        optimizers =  [mcmc],
                        significativity_tc=0.3,
                        rerun = True,
                        nb_tc = 10,
                        save=False,
                        save_plot=False)
mcmc.visualize_convergence()