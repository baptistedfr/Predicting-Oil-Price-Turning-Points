from GQLib.Optimizers import MPGA, PSO, SGA, SA, NELDER_MEAD, TABU, FA
from GQLib.Models import LPPL, LPPLS
from GQLib.enums import InputType
from GQLib.AssetProcessor import AssetProcessor
from GQLib.logging import configure_logger
import plotly.io as pio

# Configuration de Plotly pour utiliser le renderer 'browser'
pio.renderers.default = 'browser'

configure_logger("INFO")

wti = AssetProcessor(input_type = InputType.WTI)

wti.compare_optimizers(frequency = "daily",
                            optimizers =  [SA(LPPL), PSO(LPPL), MPGA(LPPL),SGA(LPPL), TABU(LPPL), FA(LPPL), NELDER_MEAD(LPPLS)],
                            significativity_tc=0.3,
                            rerun = False,
                            nb_tc = 10,
                            save=False,
                            save_plot=False,
                            save_metrics=True)

# wti = AssetProcessor(input_type = InputType.SP500)

# wti.compare_optimizers(frequency = "daily",
#                             optimizers =  [SA(LPPL), PSO(LPPL), MPGA(LPPL),SGA(LPPL), TABU(LPPL), FA(LPPL), NELDER_MEAD(LPPLS)],
#                             significativity_tc=0.3,
#                             rerun = False,
#                             nb_tc = 10,
#                             save=False,
#                             save_plot=False)

# wti = AssetProcessor(input_type = InputType.EURUSD)

# wti.compare_optimizers(frequency = "daily",
#                             optimizers =  [SA(LPPL), PSO(LPPL), MPGA(LPPL),SGA(LPPL), TABU(LPPL), FA(LPPL), NELDER_MEAD(LPPLS)],
#                             significativity_tc=0.3,
#                             rerun = False,
#                             nb_tc = 10,
#                             save=False,
#                             save_plot=False)

# wti = AssetProcessor(input_type = InputType.SSE)

# wti.compare_optimizers(frequency = "daily",
#                             optimizers =  [SA(LPPL), PSO(LPPL), MPGA(LPPL),SGA(LPPL), TABU(LPPL), FA(LPPL), NELDER_MEAD(LPPLS)],
#                             significativity_tc=0.3,
#                             rerun = False,
#                             nb_tc = 10,
#                             save=False,
#                             save_plot=False)

