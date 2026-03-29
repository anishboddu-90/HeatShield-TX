from data_cleaning import clean_data, compute_annual_stats
from simulation import run_monte_carlo
from finance import calculate_efficiency_factor, estimate_cooling_costs
from risk_engine import calculate_risk_score
import pandas as pd

# Load and clean data
df = pd.read_csv('../data/raw/mckinney.csv', skiprows=1)
df = clean_data(df)
# Set input bills and month before selecting monthly series
LAST_SUMMER_BILL = 500
LAST_SUMMER_MONTH = 7
LAST_SUMMER_YEAR = 2023
LAST_MAY_BILL = 400  # set to numeric baseline (May bill)
CONSRUCTION_YEAR = 1995
HOUSE_TYPE = 'detached'

# Use historical CDDs for the target month (simulate that month's CDD distribution)
monthly_by_year = df.groupby(['year', 'month'])['cdd'].sum()
try:
	month_series = monthly_by_year.xs(LAST_SUMMER_MONTH, level=1)
except Exception:
	# fallback: select values where month equals LAST_SUMMER_MONTH
	month_series = monthly_by_year[monthly_by_year.index.get_level_values(1) == LAST_SUMMER_MONTH]

# Run Monte Carlo on historical monthly CDDs for the target month
sim_cdds, mean_cdd, p90_cdd = run_monte_carlo(month_series)

HOUSE_SQFT = 2000
NORMALIZE_EF = True
# Calibrate efficiency using the May baseline (subtract May bill first)
efficiency_factor = calculate_efficiency_factor(
	last_bill=LAST_SUMMER_BILL,
	bill_month=LAST_SUMMER_MONTH,
	bill_year=LAST_SUMMER_YEAR,
	baseline_bill=LAST_MAY_BILL,
	baseline_bill_month=5,
	baseline_bill_year=LAST_SUMMER_YEAR,
	house_size_sqft=HOUSE_SQFT if NORMALIZE_EF else None,
	normalized=NORMALIZE_EF,
	CONSTRUCTION_YEAR=CONSRUCTION_YEAR,
	monthly_series=monthly_by_year,
)


# If EF was normalized per sqft, scale it back to $/CDD for this house before estimating costs
if NORMALIZE_EF:
	ef_for_cost = efficiency_factor * HOUSE_SQFT
else:
	ef_for_cost = efficiency_factor

# Simulated costs = simulated CDDs * (dollars per CDD for this house) + base load
sim_costs = estimate_cooling_costs(sim_cdds, ef_for_cost, base_load=LAST_MAY_BILL)

# Summary statistics in dollars (include base load in final reported costs)
mean_cost = mean_cdd * ef_for_cost + (LAST_MAY_BILL or 0)
p90_cost = p90_cdd * ef_for_cost + (LAST_MAY_BILL or 0)

user_budget = 500
risk_score = calculate_risk_score(sim_costs, user_budget)

if NORMALIZE_EF:
	# efficiency_factor is $/(CDD·sqft) — show normalized and house-level EF
	print(f"Normalized Efficiency Factor ($/(CDD·sqft)): {efficiency_factor:.6f}")
	print(f"Your House Efficiency Factor ($/CDD): {ef_for_cost:.4f}")
else:
	print(f"House Efficiency Factor ($/CDD): {efficiency_factor:.4f}")

print(f"90% Safety Cost Estimate: ${p90_cost:.2f}")
print(f"Your budget: ${user_budget:.2f}")
print(f"Risk of exceeding budget: {risk_score:.1f}%")