import pandas as pd
import matplotlib.pyplot as plt
from data_cleaning import clean_data, compute_annual_stats
import numpy as np
from simulation import run_monte_carlo

# Load and clean data
df = pd.read_csv('../data/raw/mckinney.csv', skiprows=1)
df = clean_data(df)

# Historical annual CDDs
annual_cdd = compute_annual_stats(df)

# Run Monte Carlo on historical annual CDDs
sim_cdds, mean_cdd, p90_cdd = run_monte_carlo(annual_cdd)

plt.hist(sim_cdds, bins=30, alpha=0.7, color='blue', edgecolor='black')
plt.axvline(mean_cdd, color='red', linestyle='dashed', linewidth=2, label=f'Mean CDD: {mean_cdd:.1f}')
plt.axvline(p90_cdd, color='green', linestyle='dashed', linewidth=2, label=f'90th Percentile CDD: {p90_cdd:.1f}')
plt.title('Monte Carlo Simulation of Annual Cooling Degree Days (CDDs)')
plt.xlabel('Annual CDDs')
plt.ylabel('Frequency')
plt.legend()
plt.grid()
plt.show()