import pandas as pd
import numpy as np


def run_monte_carlo(annual_data, num_simulations=10000, seed=42):
    history = np.asarray(annual_data)
    rng = np.random.default_rng(seed)

    sim_results = rng.choice(history, size=num_simulations, replace=True)

    sample_mean = np.mean(sim_results)
    sample_90th_percentile = np.percentile(sim_results, 90)
    return sim_results, sample_mean, sample_90th_percentile