from Utilities import load_data, select_sample
from MPGA import MPGA
import json

# Exemple d'utilisation
file_path = 'WTI_Spot_Price_daily.csv'
data = load_data(file_path)

time_start = "04/01/2003"  # date de début (exemple)
time_end = "04/02/2008"    # date de fin (exemple)

sample = select_sample(data, time_start, time_end)

print("Start fitting the data...")
mpga = MPGA(sample, "daily")

results = mpga.fit()

# Save results to a JSON file
with open("results-2003_2008_1.json", "w") as f:
    json.dump(results, f, indent=4)

print("Results saved to results-2003_2008.json")

time_start = "02/01/2007"  # date de début (exemple)
time_end = "02/01/2011"    # date de fin (exemple)

sample = select_sample(data, time_start, time_end)

print("Start fitting the data...")
mpga = MPGA(sample, "daily")

results = mpga.fit()

# Save results to a JSON file
with open("results-2007_2011_1.json", "w") as f:
    json.dump(results, f, indent=4)

print("Results saved to results-2007_2011.json")

time_start = "04/29/2011"  # date de début (exemple)
time_end = "08/01/2015"    # date de fin (exemple)

print("Start fitting the data...")
sample = select_sample(data, time_start, time_end)

mpga = MPGA(sample, "daily")

results = mpga.fit()

# Save results to a JSON file
with open("results-2011_2015_1.json", "w") as f:
    json.dump(results, f, indent=4)

print("Results saved to results-2011_2015.json")