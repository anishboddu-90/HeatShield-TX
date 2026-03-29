import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Ensure src/ is on the import path so we can import project modules
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(ROOT, 'src')
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from data_cleaning import clean_data
from simulation import run_monte_carlo
from finance import calculate_efficiency_factor, estimate_cooling_costs
from risk_engine import calculate_risk_score

@st.cache_data
def load_clean_data(path=None):
    # Resolve path relative to this file's directory so Streamlit can be
    # launched from any working directory.
    if path is None:
        path = os.path.join(ROOT, 'data', 'raw', 'mckinney.csv')
    elif not os.path.isabs(path):
        path = os.path.join(ROOT, path)
    df = pd.read_csv(path, skiprows=1)
    return clean_data(df)

st.set_page_config(page_title="HeatShield Collin County", layout="wide")
st.title("☀️ HeatShield: Energy Risk Portal")
st.markdown("Predicting financial vulnerability to Texas heatwaves")

# single sidebar defined later (after loading data)
# Load data early so we can populate sensible UI choices (years, months)
df = load_clean_data()
available_years = sorted(df['year'].unique().tolist()) if 'year' in df.columns else []
month_names = [
    'January','February','March','April','May','June',
    'July','August','September','October','November','December'
]

with st.sidebar:
    st.header("House Profile")
    sq_ft = st.number_input("Square Footage", value=2000, min_value=100, step=50)
    year_built = st.number_input("Year Built", value=2005, min_value=1800, max_value=2100)
    house_type = st.selectbox("House Type", options=["Detached", "Townhouse", "Apartment"], index=0, key='house_type')

    st.header("💰 Billing Calibration")
    may_bill = st.number_input(
        "May baseline bill (non-AC baseline)",
        value=110.0,
        min_value=0.0,
        step=1.0,
        help="Enter your approximate May bill representing non-AC loads (fridge, lights, etc.)."
        , disabled=st.session_state.get('simulate_profile', False)
    )
    summer_bill = st.number_input("Summer Bill (Peak)", value=400.0, min_value=0.0, step=1.0,disabled=st.session_state.get('simulate_profile', False))

    # Month selector shows names; we'll map to numeric month when used
    default_month_index = 7  # August
    summer_month_name = st.selectbox("Summer Month (for calibration)", options=month_names, index=default_month_index)
    # Year selector populated from data; fallback to most recent year
    if available_years:
        default_year_index = len(available_years) - 1
        calibration_year = st.selectbox("Calibration Year", options=available_years, index=default_year_index)
    else:
        calibration_year = st.number_input("Calibration Year", value=2023)

    # synchronized slider + textbox for budget
    if 'budget_slider' not in st.session_state:
        st.session_state['budget_slider'] = 450
    if 'budget_input' not in st.session_state:
        st.session_state['budget_input'] = 450

    def _sync_slider():
        st.session_state['budget_input'] = st.session_state['budget_slider']

    def _sync_input():
        st.session_state['budget_slider'] = st.session_state['budget_input']

    col_budget, col_budget_input = st.columns([3, 1])
    with col_budget:
        st.slider("Monthly Summer Budget ($)", 0, 2000, key='budget_slider', on_change=_sync_slider)
    with col_budget_input:
        st.number_input("", min_value=0, max_value=2000, step=1, key='budget_input', on_change=_sync_input)
    budget = st.session_state['budget_slider']

    # Monte Carlo parameters are fixed for production-like runs
    normalize = st.checkbox("Normalize EF per sqft (optional)", value=False, key='normalize')
    simulate_profile = st.checkbox("I don't have bills — simulate EF from house profile", value=False, key='simulate_profile')
    # Simulation inputs
    if simulate_profile:
        electricity_rate = st.number_input(
            "Electricity rate ($/kWh)", value=0.14, min_value=0.01, step=0.01,
            help="Your retail electricity rate. Defaults to TX average of $0.14/kWh.", key='electricity_rate')
        thermostat_temp = st.number_input(
            "Typical summer thermostat setpoint (°F)", value=74, min_value=60, max_value=82, step=1,
            help="Used to adjust consumption; cooler setpoints increase cooling energy.", key='thermostat_temp')
        hvac_seer = st.number_input(
            "HVAC SEER rating (approx)", value=14, min_value=8, max_value=25, step=1,
            help="Approximate SEER efficiency of your AC unit; higher SEER = lower energy.", key='hvac_seer')
    else:
        electricity_rate = 0.14
        thermostat_temp = 74
        hvac_seer = 14

# map selected month name to numeric month
summer_month = month_names.index(summer_month_name) + 1

# Build monthly historical CDD series for the chosen calibration month
monthly_by_year = df.groupby(['year', 'month'])['cdd'].sum()
try:
    month_series = monthly_by_year.xs(summer_month, level=1)
except Exception:
    month_series = monthly_by_year[monthly_by_year.index.get_level_values(1) == summer_month]

# Compute efficiency factor (calibration)
if simulate_profile:
    ef = calculate_efficiency_factor(
        last_bill=None,
        bill_month=summer_month,
        bill_year=calibration_year,
        baseline_bill=None,
        house_size_sqft=sq_ft if normalize else None,
        normalized=normalize,
        CONSTRUCTION_YEAR=year_built,
        HOUSE_TYPE=house_type,
        monthly_series=monthly_by_year,
        simulate_if_missing=True,
        electricity_rate=electricity_rate,
        thermostat_temp=thermostat_temp,
        hvac_seer=hvac_seer,
    )
else:
    ef = calculate_efficiency_factor(
        last_bill=summer_bill,
        bill_month=summer_month,
        bill_year=calibration_year,
        baseline_bill=may_bill,
        baseline_bill_month=5,
        baseline_bill_year=calibration_year,
        house_size_sqft=sq_ft if normalize else None,
        normalized=normalize,
        CONSTRUCTION_YEAR=year_built,
        HOUSE_TYPE=house_type,
        monthly_series=monthly_by_year,
    )

# If normalized, scale back to $/CDD for this house
if normalize:
    ef_house = ef * sq_ft
else:
    ef_house = ef

# Run Monte Carlo on historical monthly CDDs (fixed production-like draws)
SIMS = 10000
sim_cdds, mean_cdd, p90_cdd = run_monte_carlo(month_series, num_simulations=SIMS)

# Compute simulated total bills (cooling portion + baseline)
# If we're in simulation mode (no user bills), don't add the May baseline
base_load = 0.0 if simulate_profile else may_bill
sim_costs = estimate_cooling_costs(sim_cdds, ef_house, base_load=base_load)

# Summary stats
mean_cost = np.mean(sim_costs)
p90_cost = np.percentile(sim_costs, 90)

risk_percent = calculate_risk_score(sim_costs, budget)

# --- Output ---
st.subheader("📊 Your Simulated Summers")

# Display key percentiles and simpler visuals for user clarity
percentiles = [50, 75, 90, 95]
pct_vals_total = np.percentile(sim_costs, percentiles)
# compute cooling-only values relative to the selected base_load (May baseline unless simulation mode)
cooling_vals = sim_costs - base_load
# if any cooling_vals are negative because baseline > total, clamp for display
cooling_vals_clamped = np.maximum(0.0, cooling_vals)
pct_vals_cooling = np.percentile(cooling_vals_clamped, percentiles)

pct_df = pd.DataFrame({
    'percentile': [f"p{p}" for p in percentiles],
    'total_bill': pct_vals_total,
    'cooling_only': pct_vals_cooling,
}).set_index('percentile')

st.write("Estimated percentiles (Total bill and cooling portion):")

# Create a grouped bar chart with clear axes and labels
percentile_labels = ['Median', '75th', '90th', '95th']
vals_total = pct_vals_total
vals_cooling = pct_vals_cooling

fig, ax = plt.subplots(figsize=(8, 4))
ind = np.arange(len(percentile_labels))
width = 0.35
bars1 = ax.bar(ind - width/2, vals_total, width, label='Total bill', color='#1f77b4')
bars2 = ax.bar(ind + width/2, vals_cooling, width, label='Cooling only', color='#ff7f0e')

ax.set_xticks(ind)
ax.set_xticklabels(percentile_labels)
ax.set_ylabel('Amount ($)')
ax.set_xlabel('Percentile')
ax.set_title('Simulated monthly bill percentiles')
ax.legend()

# add value labels above bars
maxv = max(vals_total.max(), vals_cooling.max()) if hasattr(vals_total, 'max') else max(max(vals_total), max(vals_cooling))
for rect in bars1 + bars2:
    h = rect.get_height()
    ax.annotate(f'${h:,.0f}', xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=8)

st.pyplot(fig)
st.markdown(
    "The chart shows key percentile estimates for the simulated monthly bill distribution.\n"
    "- 'Cooling only' is the portion attributed to cooling (simulated CDD × EF).\n"
    "- 'Total bill' includes the May baseline (non-AC loads).\n"
    "We run a fixed Monte Carlo sampling of historical months to provide stable, production-like results."
)

st.divider()
col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Median Cooling", f"${np.percentile(cooling_vals_clamped,50):.2f}")
col_b.metric("Median Total", f"${np.percentile(sim_costs,50):.2f}")
col_c.metric("90% Cooling", f"${np.percentile(cooling_vals_clamped,90):.2f}")
col_d.metric("90% Total", f"${p90_cost:.2f}")

st.markdown("---")
col1, col2 = st.columns(2)
col1.metric("Risk Score", f"{risk_percent:.1f}%", delta="High" if risk_percent > 20 else "Low")
col2.metric("90% Safety Budget", f"${p90_cost:.2f}")

with st.expander("Calibration details"):
    st.write("Efficiency factor (normalized):", ef if normalize else "N/A")
    st.write("Efficiency factor (house $/CDD):", ef_house)
    st.write("Calibration month CDD (historical by year):")
    st.write(month_series.describe())
    if simulate_profile:
        # Show the simulation multipliers applied for transparency
        # compute the default base EF used by simulation so we can display it
        try:
            kwh_per_cdd_sqft = 0.0007
            sqft_val = float(sq_ft) if sq_ft is not None else 2000.0
            base_kwh_per_cdd = kwh_per_cdd_sqft * sqft_val
            thermostat_penalty = 1.0 + ((74.0 - float(thermostat_temp)) * 0.04)
            seer_factor = 14.0 / float(hvac_seer)
            rate = float(electricity_rate)
            default_base = base_kwh_per_cdd * rate * thermostat_penalty * seer_factor
        except Exception:
            default_base = None

        try:
            penalty = 1.0
            if year_built is not None:
                y = int(year_built)
                if y < 1980:
                    penalty = 1.45
                elif y < 2001:
                    penalty = 1.25
                elif y < 2016:
                    penalty = 1.15
        except Exception:
            penalty = 1.0

        ht_factor = 1.0
        if house_type.lower() == 'detached':
            ht_factor = 1.0
        elif house_type.lower() == 'townhouse':
            ht_factor = 0.9
        elif house_type.lower() == 'apartment':
            ht_factor = 0.7

        st.write("Simulation mode: used default base EF:", f"${default_base:.3f} per CDD")
        st.write("Construction-year multiplier:", f"{penalty:.2f}")
        st.write("House-type multiplier:", f"{ht_factor:.2f}")
        st.write("Electricity rate used ($/kWh):", f"${electricity_rate:.2f}")
        st.write("Thermostat setpoint used (°F):", f"{thermostat_temp}")
        st.write("HVAC SEER used:", f"{hvac_seer}")

if simulate_profile:
    st.caption("Notes: Simulation mode active — EF is estimated from house profile (kWh × rate × modifiers). Simulated costs are cooling-only (May baseline not added).")
else:
    st.caption("Notes: EF is calibrated as (Summer Bill - May Bill) / CDD_for_that_month. The app adds May bill back in final estimates.")
