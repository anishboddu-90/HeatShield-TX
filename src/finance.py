from data_cleaning import compute_monthly_stats, clean_data
import pandas as pd
import numpy as np
import datetime


def get_inflation_factor(baseline_year, target_year, cpi_series=None, default_annual_rate=0.03):
    try:
        baseline_year = int(baseline_year)
        target_year = int(target_year)
    except Exception:
        return 1.0

    if baseline_year == target_year:
        return 1.0

    def _annual_means(series):
        s = series.copy()
        if not isinstance(s.index, pd.DatetimeIndex):
            try:
                s.index = pd.to_datetime(s.index)
            except Exception:
                return None
        return s.resample('Y').mean().rename(lambda dt: dt.year)

    # 1) Use provided cpi_series if given
    if cpi_series is not None:
        try:
            annual = _annual_means(cpi_series)
            if annual is not None and baseline_year in annual.index and target_year in annual.index:
                return float(annual.loc[target_year] / annual.loc[baseline_year])
        except Exception:
            pass
    try:
        delta = int(target_year) - int(baseline_year)
        return (1.0 + float(default_annual_rate)) ** delta
    except Exception:
        return 1.0


monthly_by_year = None

def calculate_efficiency_factor(
    last_bill,
    bill_month,
    bill_year,
    baseline_bill=None,
    baseline_bill_month=None,
    baseline_bill_year=None,
    house_size_sqft=None,
    normalized=False,
    CONSTRUCTION_YEAR=None,
    HOUSE_TYPE='detached',
    monthly_series=None,
    simulate_if_missing=False,
    default_base_ef=None,
    electricity_rate=0.14,
    thermostat_temp=74.0,
    hvac_seer=14.0,
    kwh_per_cdd_sqft=None,
    adjust_baseline_for_inflation=False,
    inflation_cpi_series=None,
    default_annual_inflation=0.03,
):
    # allow last_bill to be None when simulating from profile
    if last_bill is not None:
        last_bill = float(last_bill)
    bill_month = int(bill_month)
    bill_year = int(bill_year)

    # Determine AC-only cost by subtracting baseline (e.g., May bill) if provided
    ac_cost = None
    if last_bill is not None:
        ac_cost = last_bill
        if baseline_bill is not None:
            try:
                baseline_bill = float(baseline_bill)
            except Exception:
                raise ValueError('baseline_bill must be numeric')
            # Optionally adjust baseline dollars from its year to the bill year
            if adjust_baseline_for_inflation and baseline_bill_year is not None:
                try:
                    factor = get_inflation_factor(baseline_bill_year, bill_year, cpi_series=inflation_cpi_series, default_annual_rate=default_annual_inflation)
                    baseline_bill = baseline_bill * factor
                except Exception:
                    # if adjustment fails, continue with raw baseline_bill
                    pass
            ac_cost = last_bill - baseline_bill

        if ac_cost is not None and ac_cost <= 0:
            return 0.0

    source = monthly_series if monthly_series is not None else monthly_by_year
    if source is None:
        raise ValueError('monthly_series must be provided to calculate_efficiency_factor')

    try:
        cdd_for_month = source.get((bill_year, bill_month), 0.0)
    except Exception:
        try:
            cdd_for_month = source.xs(bill_month, level=1)
            cdd_for_month = cdd_for_month.get(bill_year, 0.0)
        except Exception:
            cdd_for_month = 0.0

    if not np.isfinite(cdd_for_month) or cdd_for_month <= 0:
        return 0.0

    # If user provided bills, compute empirical EF
    if ac_cost is not None:
        base_ef = ac_cost / float(cdd_for_month)
        if normalized:
            if house_size_sqft is None or house_size_sqft <= 0:
                raise ValueError('house_size_sqft must be provided and > 0 when normalized=True')
            return base_ef / float(house_size_sqft)
        return base_ef

    # Otherwise, if allowed, simulate EF from profile using default_base_ef and
    if simulate_if_missing:
        if default_base_ef is None:
            # energy intensity per CDD per sqft (kWh)
            if kwh_per_cdd_sqft is None:
                kwh_per_cdd_sqft = 0.0007
            try:
                sqft = float(house_size_sqft) if house_size_sqft is not None else 2000.0
            except Exception:
                sqft = 2000.0
            base_kwh_per_cdd = float(kwh_per_cdd_sqft) * sqft

            # thermostat adjustment: assume 74 F neutral. Each degree below increases consumption by ~4%.
            try:
                thermostat_penalty = 1.0 + ((74.0 - float(thermostat_temp)) * 0.04)
            except Exception:
                thermostat_penalty = 1.0

            # SEER factor: baseline 14 SEER; higher SEER reduces energy proportional to 14/seer
            try:
                seer_factor = 14.0 / float(hvac_seer)
            except Exception:
                seer_factor = 1.0

            # electricity rate ($/kWh)
            try:
                rate = float(electricity_rate)
            except Exception:
                rate = 0.14

            # default_base_ef in $ per CDD
            default_base_ef = base_kwh_per_cdd * rate * thermostat_penalty * seer_factor

        # construction year factors (penalty >1 increases cost) — milder than before
        try:
            penalty_factor = 1.0
            if CONSTRUCTION_YEAR is not None:
                construction_year = int(CONSTRUCTION_YEAR)
                if construction_year < 1980:
                    penalty_factor = 1.25
                elif construction_year < 2001:
                    penalty_factor = 1.15
                elif construction_year < 2016:
                    penalty_factor = 1.08
        except Exception:
            penalty_factor = 1.0

        # house type factors (multiplicative)
        house_type_factor = 1.0
        try:
            if HOUSE_TYPE.lower() == 'detached':
                house_type_factor = 1.0
            elif HOUSE_TYPE.lower() == 'townhouse':
                house_type_factor = 0.95
            elif HOUSE_TYPE.lower() == 'apartment':
                house_type_factor = 0.85
        except Exception:
            house_type_factor = 1.0

        base_ef = float(default_base_ef) * penalty_factor * house_type_factor
        if normalized:
            if house_size_sqft is None or house_size_sqft <= 0:
                raise ValueError('house_size_sqft must be provided and > 0 when normalized=True')
            return base_ef / float(house_size_sqft)
        return base_ef

    # If we reach here, we couldn't compute EF because bills were missing
    raise ValueError('Insufficient inputs to compute efficiency factor; provide bills or enable simulate_if_missing')


def estimate_cooling_costs(cdd_value, efficiency_factor=0.3, base_load=None):
    base = float(base_load) if base_load is not None else 0.0
    return (cdd_value * efficiency_factor) + base