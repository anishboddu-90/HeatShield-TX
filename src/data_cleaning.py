import pandas as pd
import numpy as np
from datetime import datetime as dt

def clean_data(df):
    # Normalize column names
    rename_map = {
        'Date': 'date',
        'TAVG (Degrees Fahrenheit)': 'TAVG',
        'TMAX (Degrees Fahrenheit)': 'TMAX',
        'TMIN (Degrees Fahrenheit)': 'TMIN',
        'PRCP (Inches)': 'PRECIP',
        'SNOW (Inches)': 'SNOW',
        'SNWD (Inches)': 'SNWD'
    }
    df.rename(columns=rename_map, inplace=True)

    # Convert 'date' column to datetime and set index
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # Add month/year
    df['month'] = df.index.month
    df['year'] = df.index.year

    # Add Cooling Degree Days (CDD) for summer months
    summer_df = df[df['month'].isin([5, 6, 7, 8, 9 ])].copy()
    summer_df['TAVG_CALC'] = (summer_df['TMAX'] + summer_df['TMIN']) / 2
    summer_df['cdd'] = np.maximum(0, summer_df['TAVG_CALC'] - 65)
    return summer_df

def compute_annual_stats(df):
    # Return annual sums of CDD indexed by year
    annual_cdd = df.groupby('year')['cdd'].sum()
    return annual_cdd

def compute_monthly_stats(df):
    # Return monthly sums of CDD indexed by month
    monthly_cdd = df.groupby('month')['cdd'].sum()
    return monthly_cdd