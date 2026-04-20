"""
Main Simulation Engine
Runs all DGP × sample size × method combinations across R replications.
Uses joblib for parallelization across DGP × n cells.
"""

import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from config import CONFIG
from dgp import generate_data
from estimators import (
    ols_inference,
    robust_inference,
    _bootstrap_betas,
    _bootstrap_pvalue,
    bootstrap_percentile_ci,
    bootstrap_pivotal_ci,
)


def run_single_cell(dgp_name, n, cell_label=""):
    """
    Run all replications for one (DGP, n) combination.
    Returns a list of result dictionaries.

    Seeding is deterministic: each (dgp, n) cell gets a unique seed derived
    from its position in the config lists, avoiding Python's hash() which is
    randomized across processes.
    """
    dgp_idx = CONFIG["dgp_names"].index(dgp_name)
    n_idx = CONFIG["sample_sizes"].index(n)
    seed = CONFIG["random_seed"] * 10_000 + dgp_idx * 100 + n_idx
    rng = np.random.default_rng(seed)

    R = CONFIG["n_replications"]
    B = CONFIG["bootstrap_B"]
    alpha = CONFIG["alpha"]
    beta_0 = CONFIG["beta_0"]
    beta_1 = CONFIG["beta_1"]
    beta_1_null = CONFIG["beta_1_null"]

    results = []

    start = time.time()
    for r in range(R):
        if r % 500 == 0 and r > 0:
            elapsed = time.time() - start
            rate = r / elapsed
            remaining = (R - r) / rate
            print(f"  {cell_label} rep {r}/{R} "
                  f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

        # --- Data under true beta_1 (coverage + power) ---
        X, Y = generate_data(dgp_name, n, beta_0, beta_1, rng)

        # --- Data under null beta_1 = 0 (Type I error) ---
        X_null, Y_null = generate_data(dgp_name, n, beta_0, beta_1_null, rng)

        # ---- OLS ----
        b_hat, lo, hi, pval = ols_inference(X, Y, alpha)
        _, _, _, pval_null = ols_inference(X_null, Y_null, alpha)
        results.append({
            "dgp": dgp_name, "n": n, "method": "OLS", "rep": r,
            "beta_hat": b_hat, "ci_lower": lo, "ci_upper": hi,
            "p_value": pval, "p_value_null": pval_null,
        })

        # ---- HC3 Robust ----
        b_hat, lo, hi, pval = robust_inference(X, Y, alpha)
        _, _, _, pval_null = robust_inference(X_null, Y_null, alpha)
        results.append({
            "dgp": dgp_name, "n": n, "method": "HC3_Robust", "rep": r,
            "beta_hat": b_hat, "ci_lower": lo, "ci_upper": hi,
            "p_value": pval, "p_value_null": pval_null,
        })

        # ---- Bootstrap (sample once, apply both CI methods) ----
        # True-data bootstrap
        orig_b, boot_b = _bootstrap_betas(X, Y, B, rng)
        pval_boot = _bootstrap_pvalue(orig_b, boot_b)

        # Null-data bootstrap
        orig_b_null, boot_b_null = _bootstrap_betas(X_null, Y_null, B, rng)
        pval_boot_null = _bootstrap_pvalue(orig_b_null, boot_b_null)

        # Percentile CI
        lo_pct, hi_pct = bootstrap_percentile_ci(orig_b, boot_b, alpha)
        lo_pct_null, hi_pct_null = bootstrap_percentile_ci(orig_b_null, boot_b_null, alpha)
        results.append({
            "dgp": dgp_name, "n": n, "method": "Bootstrap_Percentile", "rep": r,
            "beta_hat": orig_b, "ci_lower": lo_pct, "ci_upper": hi_pct,
            "p_value": pval_boot, "p_value_null": pval_boot_null,
        })

        # Pivotal CI (same point estimate and p-value, different interval)
        lo_piv, hi_piv = bootstrap_pivotal_ci(orig_b, boot_b, alpha)
        lo_piv_null, hi_piv_null = bootstrap_pivotal_ci(orig_b_null, boot_b_null, alpha)
        results.append({
            "dgp": dgp_name, "n": n, "method": "Bootstrap_Pivotal", "rep": r,
            "beta_hat": orig_b, "ci_lower": lo_piv, "ci_upper": hi_piv,
            "p_value": pval_boot, "p_value_null": pval_boot_null,
        })

    elapsed = time.time() - start
    print(f"  ✓ {cell_label} done in {elapsed:.1f}s")
    return results


def run_simulation(n_jobs=-1):
    """
    Run the full simulation across all DGP × n combinations.

    Parameters
    ----------
    n_jobs : int
        Number of parallel workers. -1 = use all CPU cores.
        Set to 1 for debugging (sequential execution).
    """
    cells = []
    for dgp_name in CONFIG["dgp_names"]:
        for n in CONFIG["sample_sizes"]:
            label = f"[{dgp_name}, n={n}]"
            cells.append((dgp_name, n, label))

    total = len(cells)
    print(f"Running {total} cells × {CONFIG['n_replications']} replications each")
    print(f"Methods: {CONFIG['method_names']}")
    print(f"Bootstrap resamples per rep: {CONFIG['bootstrap_B']}")
    print(f"Parallelization: n_jobs={n_jobs}")
    print("=" * 60)

    overall_start = time.time()

    if n_jobs == 1:
        all_results = []
        for dgp_name, n, label in cells:
            print(f"\n{label}")
            results = run_single_cell(dgp_name, n, label)
            all_results.extend(results)
    else:
        cell_results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(run_single_cell)(dgp_name, n, label)
            for dgp_name, n, label in cells
        )
        all_results = [r for cell in cell_results for r in cell]

    overall_time = time.time() - overall_start
    print(f"\n{'=' * 60}")
    print(f"Total simulation time: {overall_time / 60:.1f} minutes")
    print(f"Total rows: {len(all_results):,}")

    return pd.DataFrame(all_results)


if __name__ == "__main__":
    df = run_simulation(n_jobs=-1)
    df.to_csv("results/raw_results.csv", index=False)
    print(f"\nResults saved to results/raw_results.csv")
    print(f"Shape: {df.shape}")
