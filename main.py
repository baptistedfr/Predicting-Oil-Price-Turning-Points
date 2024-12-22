from Utilities import load_data, select_sample
from MPGA import MPGA
import json

# Exemple d'utilisation
file_path = 'WTI_Spot_Price_daily.csv'
data = load_data(file_path)

time_start = "04/01/2003"  # date de début (exemple)
time_end = "11/14/2016"    # date de fin (exemple)

sample = select_sample(data, time_start, time_end)

mpga = MPGA(sample, "daily")

results = mpga.fit()

# Save results to a JSON file
with open("results.json", "w") as f:
    json.dump(results, f, indent=4)