"""
Analysis Module
Computes performance metrics from raw simulation results.
"""

import numpy as np
import pandas as pd
from config import CONFIG


def compute_metrics(df):
    """
    Aggregate raw simulation results into performance metrics.

    For each (DGP, n, method) combination, computes:
    - Coverage probability (and Monte Carlo SE)
    - Average CI width
    - Type I error rate
    - Power
    - Bias
    - MSE

    Parameters
    ----------
    df : pd.DataFrame
        Raw results from simulation.py

    Returns
    -------
    pd.DataFrame with one row per (DGP, n, method) combination
    """
    true_beta = CONFIG["beta_1"]
    alpha = CONFIG["alpha"]

    def agg_fn(group):
        R = len(group)

        # Coverage: does the CI contain the true beta_1?
        covers = (group["ci_lower"] <= true_beta) & (true_beta <= group["ci_upper"])
        coverage = covers.mean()
        mc_se = np.sqrt(coverage * (1 - coverage) / R)

        # CI width
        widths = group["ci_upper"] - group["ci_lower"]
        avg_width = widths.mean()
        median_width = widths.median()

        # Type I error: reject H0 when H0 is true (beta_1 = 0)
        type1 = (group["p_value_null"] < alpha).mean()
        mc_se_type1 = np.sqrt(type1 * (1 - type1) / R)

        # Power: reject H0 when H0 is false (true beta_1 ≠ 0)
        power = (group["p_value"] < alpha).mean()
        mc_se_power = np.sqrt(power * (1 - power) / R)

        # Bias and MSE
        beta_hats = group["beta_hat"]
        bias = beta_hats.mean() - true_beta
        mse = ((beta_hats - true_beta) ** 2).mean()
        std_dev = beta_hats.std()

        return pd.Series({
            "coverage": coverage,
            "mc_se_coverage": mc_se,
            "avg_ci_width": avg_width,
            "median_ci_width": median_width,
            "type1_error": type1,
            "mc_se_type1": mc_se_type1,
            "power": power,
            "mc_se_power": mc_se_power,
            "bias": bias,
            "mse": mse,
            "std_dev": std_dev,
            "n_reps": R,
        })

    metrics = (
        df.groupby(["dgp", "n", "method"])
        .apply(agg_fn, include_groups=False)
        .reset_index()
    )

    return metrics


def print_summary(metrics):
    """Print a formatted summary of key results."""
    print("\n" + "=" * 80)
    print("SIMULATION RESULTS SUMMARY")
    print("=" * 80)

    for dgp in CONFIG["dgp_names"]:
        dgp_data = metrics[metrics["dgp"] == dgp]
        print(f"\n{'─' * 80}")
        print(f"DGP: {dgp.upper()}")
        print(f"{'─' * 80}")

        # Coverage table
        print(f"\n  Coverage Probability (nominal = {1 - CONFIG['alpha']:.2f}):")
        pivot = dgp_data.pivot(index="method", columns="n", values="coverage")
        pivot_se = dgp_data.pivot(index="method", columns="n", values="mc_se_coverage")

        for method in pivot.index:
            row = "    " + f"{method:<12}"
            for n_val in pivot.columns:
                cov = pivot.loc[method, n_val]
                se = pivot_se.loc[method, n_val]
                row += f"  n={n_val}: {cov:.3f}±{se:.3f}"
            print(row)

        # Type I error table
        print(f"\n  Type I Error Rate (nominal = {CONFIG['alpha']:.2f}):")
        pivot_t1 = dgp_data.pivot(index="method", columns="n", values="type1_error")
        for method in pivot_t1.index:
            row = "    " + f"{method:<12}"
            for n_val in pivot_t1.columns:
                t1 = pivot_t1.loc[method, n_val]
                row += f"  n={n_val}: {t1:.3f}"
            print(row)

    print(f"\n{'=' * 80}")


if __name__ == "__main__":
    # Load and analyze saved results
    df = pd.read_csv("results/raw_results.csv")
    metrics = compute_metrics(df)
    metrics.to_csv("results/metrics.csv", index=False)
    print("Metrics saved to results/metrics.csv")
    print_summary(metrics)
