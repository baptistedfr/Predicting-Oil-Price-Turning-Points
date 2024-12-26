from Utilities import load_data, select_sample
from MPGA import MPGA
import json
import time
# Exemple d'utilisation
file_path = 'data/WTI_Spot_Price_daily.csv'
data = load_data(file_path)
start = time.time()
time_start = "04/01/2003"  # date de début (exemple)
time_end = "04/02/2008"    # date de fin (exemple)

sample = select_sample(data, time_start, time_end)

mpga = MPGA(sample, "daily")

results = mpga.fit()
print(time.time()-start)
# Save results to a JSON file
# with open("results-test.json", "w") as f:
#     json.dump(results, f, indent=4)