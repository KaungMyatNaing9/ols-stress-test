"""
Run the full simulation study end-to-end.

Usage:
    python run.py              # Full run (parallel)
    python run.py --quick      # Quick test run (R=200, sequential)
    python run.py --plots-only # Just regenerate plots from saved results
"""

import sys
import time
import os

from config import CONFIG
from simulation import run_simulation
from analysis import compute_metrics, print_summary
from visualizations import generate_all_plots


def main():
    # Parse simple command-line flags
    quick_mode = "--quick" in sys.argv
    plots_only = "--plots-only" in sys.argv

    # Ensure output directories exist
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    if plots_only:
        import pandas as pd
        print("Loading saved results...")
        metrics = pd.read_csv("results/metrics.csv")
        generate_all_plots(metrics)
        return

    if quick_mode:
        print("=" * 60)
        print("QUICK MODE: R=200, B=500 (for testing/debugging)")
        print("=" * 60)
        CONFIG["n_replications"] = 200
        CONFIG["bootstrap_B"] = 500
        n_jobs = 1  # Sequential for easier debugging
    else:
        n_jobs = -1  # Use all cores

    # ---- Step 1: Run Simulation ----
    print("\n" + "=" * 60)
    print("STEP 1: RUNNING MONTE CARLO SIMULATION")
    print("=" * 60)

    start = time.time()
    df = run_simulation(n_jobs=n_jobs)
    sim_time = time.time() - start

    df.to_csv("results/raw_results.csv", index=False)
    print(f"\nRaw results saved: results/raw_results.csv ({len(df):,} rows)")
    print(f"Simulation time: {sim_time / 60:.1f} minutes")

    # ---- Step 2: Compute Metrics ----
    print("\n" + "=" * 60)
    print("STEP 2: COMPUTING PERFORMANCE METRICS")
    print("=" * 60)

    metrics = compute_metrics(df)
    metrics.to_csv("results/metrics.csv", index=False)
    print(f"Metrics saved: results/metrics.csv ({len(metrics)} rows)")
    print_summary(metrics)

    # ---- Step 3: Generate Plots ----
    print("\n" + "=" * 60)
    print("STEP 3: GENERATING VISUALIZATIONS")
    print("=" * 60)

    generate_all_plots(metrics)

    # ---- Done ----
    total_time = time.time() - start
    print("\n" + "=" * 60)
    print(f"ALL DONE! Total time: {total_time / 60:.1f} minutes")
    print("=" * 60)
    print("\nOutputs:")
    print("  results/raw_results.csv   - Full simulation data")
    print("  results/metrics.csv       - Aggregated metrics")
    print("  figures/01_coverage_probability.png")
    print("  figures/02_type1_error.png")
    print("  figures/03_power_curves.png")
    print("  figures/04_ci_width.png")
    print("  figures/05_bias_mse.png")
    print("  figures/06_coverage_heatmap.png")


if __name__ == "__main__":
    main()
