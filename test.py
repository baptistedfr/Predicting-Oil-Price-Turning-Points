from GQLib.Framework import Framework

fw = Framework("daily")

best_results = fw.analyze(result_json_name="results/SGA/daily/LPPL-results_daily_01-04-03 02-01-08_SGA.json")
best_results_2 = fw.analyze(result_json_name="results/PSO/daily/LPPL-results_daily_01-04-03 02-01-08_PSO.json")
best_results_3 = fw.analyze(result_json_name="results/MPGA/daily/LPPL-results_daily_01-04-03 02-01-08_MPGA.json")

fw.compare_results_rectangle(multiple_results={"SGA": best_results, "PSO": best_results_2, 'MPGA': best_results_3},
                   name="Predicted LPPL critical times", data_name="WTI Data", real_tc= "15/07/2008",
                   start_date="04/01/2003", end_date="04/02/2008")
