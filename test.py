from GQLib.Framework import Framework
from GQLib.Optimizers import MPGA, PSO, SGA, SA, NELDER_MEAD, TABU, FA
from GQLib.Models import LPPL, LPPLS
from GQLib.utilities import generate_all_dates, generate_all_rectangle

generate_all_rectangle(frequency = "daily",
                        lppl_model = LPPL,
                        optimizers = [SA(), SGA(), NELDER_MEAD(), TABU(), FA(), MPGA(), PSO()],
                        significativity_tc=0.3,
                        rerun = True,
                        save=True)

generate_all_rectangle(frequency = "daily",
                        lppl_model = LPPLS,
                        optimizers = [SA(LPPLS), SGA(LPPLS), NELDER_MEAD(LPPLS), TABU(LPPLS), MPGA(LPPLS), PSO(LPPLS)],
                        significativity_tc=0.3,
                        rerun = True,
                        save=True)

generate_all_rectangle(frequency = "weekly",
                        lppl_model = LPPL,
                        optimizers = [SA(), SGA(), NELDER_MEAD(), TABU(), MPGA(), PSO()],
                        significativity_tc=0.3,
                        rerun = True,
                        save=True)

generate_all_rectangle(frequency = "weekly",
                        lppl_model = LPPLS,
                        optimizers = [SA(LPPLS), SGA(LPPLS), NELDER_MEAD(LPPLS), TABU(LPPLS), MPGA(LPPLS), PSO(LPPLS)],
                        significativity_tc=0.3,
                        rerun = True,
                        save=True)
# fw_daily = Framework("daily", lppl_model=LPPL)
# fw_daily.generate_all_rectangle(optimizers = [SA(), SGA(), NELDER_MEAD(), TABU()] , rerun=True, save=True)
# fw_weekly = Framework("weekly", lppl_model=LPPL)
# fw_weekly.generate_all_rectangle(optimizers = [SA(), SGA(), NELDER_MEAD(), TABU()] , rerun=True, save=True)
# best_results = fw.analyze(result_json_name="results/SGA/daily/04-2003 01-2008.json")
# best_results_2 = fw.analyze(result_json_name="results/PSO/daily/04-2003 01-2008.json")
# best_results_3 = fw.analyze(result_json_name="results/MPGA/daily/04-2003 01-2008.json")
# best_results_4 = fw.analyze(result_json_name="results/SA/daily/04-2003 01-2008.json")
#
# fw.compare_results_rectangle(multiple_results={"SGA": best_results, "PSO": best_results_2, 'MPGA': best_results_3, "SA": best_results_4},
#                    name="Predicted LPPL critical times", data_name="WTI Data", real_tc= "03/07/2008",
#                    start_date="04/01/2003", end_date="04/02/2008")
