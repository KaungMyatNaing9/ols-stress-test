"""
Data-Generating Processes (DGPs)
Each function generates (X, Y) pairs from a known linear model
with different error structures.
"""

import numpy as np


def generate_data(dgp_name: str, n: int, beta_0: float, beta_1: float, rng: np.random.Generator):
    """
    Generate (X, Y) from the specified DGP.

    Parameters
    ----------
    dgp_name : str
        One of: 'normal', 'heavy_tails', 'heteroskedastic', 'outliers', 'skewed'
    n : int
        Sample size
    beta_0 : float
        True intercept
    beta_1 : float
        True slope (the parameter we're testing)
    rng : np.random.Generator
        Random number generator for reproducibility

    Returns
    -------
    X : ndarray of shape (n,)
    Y : ndarray of shape (n,)
    """
    X = rng.standard_normal(n)

    if dgp_name == "normal":
        # Baseline: iid N(0,1) errors — textbook assumptions satisfied
        eps = rng.standard_normal(n)

    elif dgp_name == "heavy_tails":
        # t-distribution with df=3 (excess kurtosis ≈ ∞ for df≤4)
        # Rescale so Var(eps) = 1 for fair comparison
        # Var(t_df) = df/(df-2) = 3 for df=3
        eps = rng.standard_t(df=3, size=n) / np.sqrt(3)

    elif dgp_name == "heteroskedastic":
        # Variance grows with |X|: sigma_i = 0.5 * |X_i| + 0.1
        # Added 0.1 floor to avoid near-zero variance when X ≈ 0
        sigma = 0.5 * np.abs(X) + 0.1
        eps = rng.normal(0, sigma)

    elif dgp_name == "outliers":
        # Contaminated normal: 95% from N(0,1), 5% from N(0,25)
        # Simulates data with occasional wild observations
        eps = rng.standard_normal(n)
        contamination_mask = rng.random(n) < 0.05
        eps[contamination_mask] = rng.normal(0, 5, size=contamination_mask.sum())

    elif dgp_name == "skewed":
        # Exponential(1) shifted to mean 0
        # Skewness = 2, so strongly asymmetric
        eps = rng.exponential(1.0, size=n) - 1.0

    else:
        raise ValueError(f"Unknown DGP: {dgp_name}")

    Y = beta_0 + beta_1 * X + eps
    return X, Y
