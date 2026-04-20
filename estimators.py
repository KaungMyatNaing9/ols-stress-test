"""
Inference Methods for Linear Regression
Each public function returns: (beta_hat, ci_lower, ci_upper, p_value) for beta_1.
"""

import numpy as np
import statsmodels.api as sm


def ols_inference(X, Y, alpha=0.05):
    """Standard OLS with classical (homoskedastic) standard errors."""
    X_const = sm.add_constant(X)
    model = sm.OLS(Y, X_const).fit()
    ci = model.conf_int(alpha=alpha)
    return model.params[1], ci[1, 0], ci[1, 1], model.pvalues[1]


def robust_inference(X, Y, alpha=0.05):
    """
    OLS with HC3 heteroskedasticity-consistent standard errors.
    HC3 applies a leverage-based correction that improves small-sample accuracy
    over HC0/HC1.
    """
    X_const = sm.add_constant(X)
    model = sm.OLS(Y, X_const).fit(cov_type="HC3")
    ci = model.conf_int(alpha=alpha)
    return model.params[1], ci[1, 0], ci[1, 1], model.pvalues[1]


# ---------------------------------------------------------------------------
# Bootstrap internals (shared by both CI methods)
# ---------------------------------------------------------------------------

def _bootstrap_betas(X, Y, B, rng):
    """
    Fast numpy bootstrap: resample (X_i, Y_i) pairs B times and refit OLS.

    Each resample computes its own (Xb'Xb)^{-1} Xb'Yb — correct because
    each bootstrap draw has a different design matrix.

    Returns
    -------
    original_beta : float
        OLS estimate from the original data.
    boot_betas : ndarray
        Bootstrap distribution of the slope estimate (NaNs removed).
    """
    n = len(X)
    Xm = np.column_stack([np.ones(n), X])

    # Original OLS (no statsmodels needed)
    original_beta = np.linalg.solve(Xm.T @ Xm, Xm.T @ Y)[1]

    boot_betas = np.empty(B)
    for b in range(B):
        idx = rng.choice(n, size=n, replace=True)
        Xb, Yb = Xm[idx], Y[idx]
        try:
            boot_betas[b] = np.linalg.solve(Xb.T @ Xb, Xb.T @ Yb)[1]
        except np.linalg.LinAlgError:
            boot_betas[b] = np.nan

    return original_beta, boot_betas[~np.isnan(boot_betas)]


def _bootstrap_pvalue(original_beta, boot_betas):
    """
    Two-sided bootstrap p-value for H0: beta_1 = 0.

    Uses the shift (pivot) method: center the bootstrap distribution at
    original_beta to approximate the null distribution, then ask how often
    that null distribution exceeds |original_beta|.

    Centering at original_beta (not mean(boot_betas)) is the correct pivot —
    the two differ by the bootstrap bias, which matters in small samples.
    """
    boot_shifted = boot_betas - original_beta
    return float(np.mean(np.abs(boot_shifted) >= np.abs(original_beta)))


def bootstrap_percentile_ci(original_beta, boot_betas, alpha):
    """
    Percentile CI: use the empirical quantiles of the bootstrap distribution directly.

    Simple but can under-cover when the bootstrap distribution is skewed or biased
    relative to the true sampling distribution.
    """
    lo = float(np.percentile(boot_betas, 100 * alpha / 2))
    hi = float(np.percentile(boot_betas, 100 * (1 - alpha / 2)))
    return lo, hi


def bootstrap_pivotal_ci(original_beta, boot_betas, alpha):
    """
    Pivotal (basic/reflected) CI: 2*beta_hat - quantile(boot_betas).

    Corrects for bootstrap bias by reflecting the distribution around the
    original estimate. Tends to have better coverage than percentile CI
    when the sampling distribution is skewed or the estimator is biased.
    """
    lo = float(2 * original_beta - np.percentile(boot_betas, 100 * (1 - alpha / 2)))
    hi = float(2 * original_beta - np.percentile(boot_betas, 100 * alpha / 2))
    return lo, hi


# ---------------------------------------------------------------------------
# Public bootstrap inference wrappers (match the (beta_hat, lo, hi, pval) API)
# ---------------------------------------------------------------------------

def bootstrap_percentile_inference(X, Y, B=2000, alpha=0.05, rng=None):
    """Paired bootstrap with percentile CI."""
    original_beta, boot_betas = _bootstrap_betas(X, Y, B, rng)
    lo, hi = bootstrap_percentile_ci(original_beta, boot_betas, alpha)
    p_value = _bootstrap_pvalue(original_beta, boot_betas)
    return original_beta, lo, hi, p_value


def bootstrap_pivotal_inference(X, Y, B=2000, alpha=0.05, rng=None):
    """Paired bootstrap with pivotal (basic/reflected) CI."""
    original_beta, boot_betas = _bootstrap_betas(X, Y, B, rng)
    lo, hi = bootstrap_pivotal_ci(original_beta, boot_betas, alpha)
    p_value = _bootstrap_pvalue(original_beta, boot_betas)
    return original_beta, lo, hi, p_value
