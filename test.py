from GQLib.Framework import Framework
from GQLib.Optimizers import MPGA, PSO, SGA, SA, NELDER_MEAD, TABU, FA
from GQLib.Models import LPPL, LPPLS
from GQLib.utilities import generate_all_dates, generate_all_rectangle

# generate_all_rectangle(frequency = "daily",
#                         optimizers = [SA(LPPL), SGA(LPPL), NELDER_MEAD(LPPLS), TABU(LPPL), FA(LPPL), MPGA(LPPL), PSO(LPPL)],
#                         significativity_tc=0.3,
#                         rerun = True,
#                         nb_tc = 20,
#                         save=True,
#                         save_plot=True)

# generate_all_rectangle(frequency = "daily",
#                         optimizers = [SA(LPPLS), SGA(LPPLS), NELDER_MEAD(LPPLS), TABU(LPPLS), FA(LPPLS), MPGA(LPPLS), PSO(LPPLS)],
#                         significativity_tc=0.3,
#                         rerun = True,
#                         nb_tc = 20,
#                         save=True,
#                          save_plot=True)

generate_all_rectangle(frequency = "weekly",
                        optimizers = [SA(LPPL), SGA(LPPL), NELDER_MEAD(LPPLS)],
                        significativity_tc=0.3,
                        rerun = True,
                        nb_tc = 20,
                        save=True,
                        save_plot=True)

generate_all_rectangle(frequency = "weekly",
                        optimizers = [SA(LPPLS), SGA(LPPLS), NELDER_MEAD(LPPLS), TABU(LPPLS), FA(LPPLS), MPGA(LPPLS), PSO(LPPLS)],
                        significativity_tc=0.3,
                        rerun = True,
                        nb_tc = 20,
                        save=True,
                        save_plot=True)

