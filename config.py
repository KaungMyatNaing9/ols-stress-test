"""
Configuration for Linear Regression Inference Simulation Study
All parameters in one place for easy tuning.
"""

CONFIG = {
    # True regression coefficients: Y = beta_0 + beta_1 * X + eps
    "beta_0": 2,
    # beta_1 = 0.5 chosen so power varies meaningfully across n=30..500.
    # With X~N(0,1) and eps~N(0,1), SE(beta_hat) ≈ 1/sqrt(n), giving
    # t-stat ≈ 0.5*sqrt(n): power ≈ 45% at n=30, ~99% at n=500.
    "beta_1": 0.5,
    "beta_1_null": 0,  # For Type I error simulations

    # Sample sizes to test (small → large)
    "sample_sizes": [30, 50, 100, 250, 500],

    # Simulation settings
    "n_replications": 5_000,  # Start with 5k; increase to 10k for final run
    "bootstrap_B": 2_000,     # Bootstrap resamples per replication
    "alpha": 0.05,            # Significance level
    "random_seed": 42,

    # Data-generating processes
    "dgp_names": [
        "normal",           # Baseline: N(0,1) errors
        "heavy_tails",      # t(df=3) errors — mimics stock returns
        "heteroskedastic",  # Variance scales with |X| — mimics housing prices
        "outliers",         # 95% N(0,1) + 5% N(0,25) — mimics clinical data
        "skewed",           # Shifted exponential — mimics insurance claims
    ],

    # Inference methods (order determines plot legend order)
    "method_names": [
        "OLS",
        "HC3_Robust",
        "Bootstrap_Percentile",
        "Bootstrap_Pivotal",
    ],

    # DGP display labels for plots
    "dgp_labels": {
        "normal": "Normal Errors\n(Baseline)",
        "heavy_tails": "Heavy Tails\n(Stock Returns)",
        "heteroskedastic": "Heteroskedastic\n(Housing Prices)",
        "outliers": "Outliers\n(Clinical Trials)",
        "skewed": "Skewed Errors\n(Insurance Claims)",
    },

    # Method display labels for plots
    "method_labels": {
        "OLS": "OLS",
        "HC3_Robust": "HC3 Robust",
        "Bootstrap_Percentile": "Bootstrap (Percentile)",
        "Bootstrap_Pivotal": "Bootstrap (Pivotal)",
    },
}
