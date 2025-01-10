from GQLib.Framework import Framework

fw = Framework("daily")
fw.generate_all_rectangle(rerun=True, save=True)

# best_results = fw.analyze(result_json_name="results/SGA/daily/04-2003 01-2008.json")
# best_results_2 = fw.analyze(result_json_name="results/PSO/daily/04-2003 01-2008.json")
# best_results_3 = fw.analyze(result_json_name="results/MPGA/daily/04-2003 01-2008.json")
# best_results_4 = fw.analyze(result_json_name="results/SA/daily/04-2003 01-2008.json")
#
# fw.compare_results_rectangle(multiple_results={"SGA": best_results, "PSO": best_results_2, 'MPGA': best_results_3, "SA": best_results_4},
#                    name="Predicted LPPL critical times", data_name="WTI Data", real_tc= "03/07/2008",
#                    start_date="04/01/2003", end_date="04/02/2008")
