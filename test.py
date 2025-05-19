from GQLib.Optimizers import MPGA, PSO, SGA, SA, NELDER_MEAD, TABU, FA
from GQLib.Models import LPPL, LPPLS
from GQLib.enums import InputType
from GQLib.AssetProcessor import AssetProcessor
from GQLib.logging import configure_logger

configure_logger("INFO")

wti = AssetProcessor(input_type = InputType.WTI)

wti.compare_optimizers(frequency = "daily",
                        optimizers =  [NELDER_MEAD(LPPLS), SA(LPPLS)],
                        significativity_tc=0.3,
                        rerun = False,
                        nb_tc = 10,
                        save=False,
                        save_plot=False)