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
st.title("HeatShield Energy Risk Portal")
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


    st.header("Billing Calibration")
    simulate_profile = st.session_state.get('simulate_profile', False)
    if not simulate_profile:
        may_bill = st.number_input(
            "May baseline bill (non-AC baseline)",
            value=110.0,
            min_value=0.0,
            step=1.0,
            help="Enter your approximate May bill representing non-AC loads (fridge, lights, etc.)."
        )
        summer_bill = st.number_input("Summer Bill (Peak)", value=400.0, min_value=0.0, step=1.0)
    else:
        st.info("Simulation mode: billing inputs are hidden — results use simulated EF and cooling-only costs.")
        may_bill = None
        summer_bill = None

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
        st.number_input("Monthly budget (hidden label)", min_value=0, max_value=2000, step=1, key='budget_input', on_change=_sync_input, label_visibility='collapsed')
    budget = st.session_state['budget_slider']

    # Monte Carlo parameters are fixed for production-like runs
    normalize = st.checkbox(
        "Normalize EF per sqft (optional)",
        value=False,
        key='normalize',
        help="When checked, EF is reported per square foot so you can compare homes of different sizes. EF: the $/CDD efficiency factor calibrated from your bills; higher EF means more sensitive to heat."
    )
    simulate_profile = st.checkbox(
        "I don't have bills — simulate EF from house profile",
        value=False,
        key='simulate_profile',
        help="Estimate cooling costs from house profile and user inputs (rate, thermostat, SEER). Billing inputs will be disabled."
    )
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
        # power-user tuning knob for the base kWh-per-CDD-per-sqft constant
        kwh_per_cdd_sqft = st.number_input(
            "Base kWh per CDD per sqft (power-user)",
            value=0.0007,
            min_value=0.0001,
            max_value=0.0020,
            step=0.0001,
            format="%.4f",
            help="Tunable constant for energy intensity per CDD per sqft. Default 0.0007 kWh/CDD/sqft.")
    else:
        electricity_rate = 0.14
        thermostat_temp = 74
        hvac_seer = 14
        kwh_per_cdd_sqft = 0.0007
    # Mitigation slider: hypothetical percent efficiency improvement (0-30%)
    mitigation_pct = st.slider(
        "Efficiency Improvement (e.g., New Insulation/AC)",
        min_value=0,
        max_value=30,
        value=0,
        step=1,
        help="Apply a hypothetical efficiency improvement (reduces effective $/CDD).",
        key='mitigation_pct'
    )

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
        kwh_per_cdd_sqft=kwh_per_cdd_sqft,
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

# Apply mitigation improvement to the house EF used in simulations
try:
    mitigation_fraction = float(st.session_state.get('mitigation_pct', mitigation_pct)) / 100.0
except Exception:
    mitigation_fraction = 0.0
ef_house_adj = float(ef_house) * max(0.0, (1.0 - mitigation_fraction))

# Run Monte Carlo on historical monthly CDDs (fixed production-like draws)
SIMS = 10000
sim_cdds, mean_cdd, p90_cdd = run_monte_carlo(month_series, num_simulations=SIMS)

# Compute simulated total bills (cooling portion + baseline)
# If we're in simulation mode (no user bills), don't add the May baseline
base_load = 0.0 if simulate_profile else may_bill
sim_costs = estimate_cooling_costs(sim_cdds, ef_house_adj, base_load=base_load)

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

# Draw a horizontal dashed budget line so users can see when percentiles cross the budget
try:
    ax.axhline(budget, color='red', linestyle='--', linewidth=1.25, label='Monthly Budget')
    # place a small label at the right edge
    ax.annotate(f'Budget ${budget}', xy=(ind[-1] + 0.6, budget), xytext=(0, 3), textcoords='offset points', color='red', ha='right', va='bottom', fontsize=9)
except Exception:
    pass

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

# Policy summary / insights block
insight_title = None
if risk_percent < 10:
    insight_title = "Low Vulnerability"
    insight_style = "success"
elif risk_percent < 20:
    insight_title = "Moderate Vulnerability"
    insight_style = "info"
else:
    insight_title = "High Vulnerability"
    insight_style = "warning"

insight_msg = (
    f"Risk Score: {risk_percent:.1f}% — {insight_title}.\n"
    "Recommended actions: "
)
if risk_percent >= 20:
    insight_msg += "Check attic insulation, schedule HVAC tune-up, or apply for county weatherization programs."
elif risk_percent >= 10:
    insight_msg += "Consider adding attic insulation or improving thermostat setbacks."
else:
    insight_msg += "Maintain current systems and monitor usage; consider small HVAC tune-ups."

if insight_style == 'success':
    st.success(insight_msg)
elif insight_style == 'info':
    st.info(insight_msg)
else:
    st.warning(insight_msg)

# Export buttons: percentiles CSV and a short summary text
import io
csv_bytes = io.StringIO()
pct_df.to_csv(csv_bytes)
csv_data = csv_bytes.getvalue().encode('utf-8')

summary_lines = []
summary_lines.append(f"Risk Score: {risk_percent:.1f}%")
summary_lines.append(f"Monthly Budget: ${budget}")
summary_lines.append(f"Mitigation applied: {mitigation_pct}%")
summary_lines.append("")
summary_lines.append("Percentiles (total / cooling-only):")
for idx in pct_df.index:
    summary_lines.append(f"{idx}: ${pct_df.loc[idx,'total_bill']:.2f} / ${pct_df.loc[idx,'cooling_only']:.2f}")
summary_text = "\n".join(summary_lines)

st.download_button("Download percentiles CSV", data=csv_data, file_name="heatshield_percentiles.csv", mime='text/csv')
st.download_button("Download summary TXT", data=summary_text, file_name="heatshield_summary.txt", mime='text/plain')
# PDF export: include title, summary and the chart image
try:
    from reportlab.lib.pagesizes import letter, landscape
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
    from reportlab.lib.styles import getSampleStyleSheet

    pdf_bytes = io.BytesIO()
    doc = SimpleDocTemplate(pdf_bytes, pagesize=landscape(letter))
    styles = getSampleStyleSheet()
    elems = []
    elems.append(Paragraph("HeatShield Simulation Summary", styles['Title']))
    elems.append(Spacer(1, 12))
    for line in summary_lines:
        elems.append(Paragraph(line, styles['Normal']))
        elems.append(Spacer(1, 6))

    # embed the matplotlib figure as an image
    try:
        img_buf = io.BytesIO()
        fig.savefig(img_buf, format='png', bbox_inches='tight')
        img_buf.seek(0)
        rl_img = RLImage(img_buf, width=700, height=250)
        elems.append(Spacer(1, 12))
        elems.append(rl_img)
    except Exception:
        pass

    doc.build(elems)
    pdf_bytes.seek(0)
    st.download_button("Download PDF report", data=pdf_bytes.getvalue(), file_name="heatshield_report.pdf", mime='application/pdf')
except Exception:
    st.info("PDF export requires `reportlab` installed. See requirements.txt.")

st.divider()
col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Median Cooling", f"${np.percentile(cooling_vals_clamped,50):.2f}")
col_b.metric("Median Total", f"${np.percentile(sim_costs,50):.2f}")
col_c.metric("90% Cooling", f"${np.percentile(cooling_vals_clamped,90):.2f}")
col_d.metric("90% Total", f"${p90_cost:.2f}")

st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    st.metric("Risk Score", f"{risk_percent:.1f}%", delta="High" if risk_percent > 20 else "Low")
    st.info("Risk score: estimated probability the monthly bill will exceed your Monthly Summer Budget (shown as dashed red line on the chart).")
with col2:
    st.metric("90% Safety Budget", f"${p90_cost:.2f}")

with st.expander("Calibration details"):
    # Friendly labels for general users
    st.write("Efficiency factor (normalized):", f"{ef:.4f}" if normalize else "N/A")
    st.write("House Heat Sensitivity:", f"${ef_house:.2f} per degree-day")
    st.write("Calibration month CDD (historical summary):")
    st.write(month_series.describe().round(2))

    if simulate_profile:
        # show the user-friendly simulation inputs
        st.write("Simulation inputs used:")
        st.write("- Electricity rate ($/kWh):", f"${electricity_rate:.2f}")
        st.write("- Thermostat setpoint (°F):", f"{thermostat_temp}")
        st.write("- HVAC SEER used:", f"{hvac_seer}")
    # show mitigation effect
    st.write("Mitigation applied:", f"{mitigation_pct}%")
    st.write("House Heat Sensitivity (adjusted):", f"${ef_house_adj:.3f} per degree-day")

    # Technical data in a collapsed expander for power users / debugging
    with st.expander("Technical Calibration Data"):
        st.write("Raw EF (normalized):", ef if normalize else "N/A")
        st.write("Raw EF (house $/CDD):", ef_house)
        st.write("Monthly CDD series (full):")
        st.write(month_series.describe())

if simulate_profile:
    st.caption("Notes: Simulation mode active — EF is estimated from house profile (kWh × rate × modifiers). Simulated costs are cooling-only (May baseline not added).")
else:
    st.caption("Notes: EF is calibrated as (Summer Bill - May Bill) / CDD_for_that_month. The app adds May bill back in final estimates.")
