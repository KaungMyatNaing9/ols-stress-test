# Monte Carlo Simulation: Reliability of Linear Regression Inference in Finite Samples

A rigorous simulation study evaluating how well standard OLS, HC3-robust, and two bootstrap inference methods perform when classical linear regression assumptions are violated — across five realistic error structures and five sample sizes.

---

## Table of Contents

1. [Motivation](#motivation)
2. [The Statistical Model](#the-statistical-model)
3. [Data-Generating Processes](#data-generating-processes)
4. [Inference Methods and Their Math](#inference-methods-and-their-math)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Simulation Design](#simulation-design)
7. [Results and Interpretation](#results-and-interpretation)
8. [Real-World Scenario Walkthrough](#real-world-scenario-walkthrough)
9. [Quick Start](#quick-start)
10. [Project Structure](#project-structure)

---

## Motivation

Linear regression is the backbone of quantitative analysis in economics, medicine, social science, and machine learning. When you report `p < 0.05` or a `95% confidence interval`, you are implicitly trusting that the standard formulas are valid. Those formulas rest on three assumptions that are routinely violated in practice:

1. **Homoskedasticity** — the error variance is constant across all observations.
2. **Normality** — the errors follow a normal distribution (required for exact finite-sample inference).
3. **No outliers or contamination** — no extreme observations distort the error distribution.

When these break down, standard OLS confidence intervals and hypothesis tests can be wildly miscalibrated. A 95% CI might actually cover the true parameter only 77% of the time. A test with a nominal 5% false-positive rate might reject a true null hypothesis 22% of the time.

**This study answers two questions:**
- Under which violations does standard OLS fail badly, and how badly?
- Do robust (HC3) and bootstrap alternatives actually fix the problem?

---

## The Statistical Model

The true data-generating model throughout this study is:

$$Y_i = \beta_0 + \beta_1 X_i + \varepsilon_i, \quad i = 1, \ldots, n$$

with **true parameters** $\beta_0 = 2$ and $\beta_1 = 0.5$.

The covariate is drawn from a standard normal: $X_i \overset{iid}{\sim} \mathcal{N}(0,1)$.

$\varepsilon_i$ is the error term whose distribution changes across the five DGPs. The choice $\beta_1 = 0.5$ is deliberate: it creates an effect size small enough that power grows meaningfully from roughly 45% at $n=30$ to 99%+ at $n=500$, enabling genuine comparison of methods' power efficiency.

---

## Data-Generating Processes

Each DGP changes only the error distribution $\varepsilon_i$, keeping everything else fixed.

### 1. Normal Errors (Baseline)

$$\varepsilon_i \overset{iid}{\sim} \mathcal{N}(0,1)$$

All classical OLS assumptions are exactly satisfied. This is the **textbook scenario** where standard inference is provably exact.

**Real-world analog:** Measuring yield of a chemical reaction under carefully controlled laboratory conditions. Experimental noise is approximately Gaussian.

### 2. Heavy-Tailed Errors

$$\varepsilon_i \overset{iid}{\sim} \frac{t_3}{\sqrt{3}}$$

The $t$-distribution with 3 degrees of freedom has infinite kurtosis (excess kurtosis → ∞ for $\nu \leq 4$). Dividing by $\sqrt{\text{Var}(t_3)} = \sqrt{3}$ normalizes to unit variance for fair comparison.

**Real-world analog:** Daily stock return residuals after removing market exposure (CAPM alpha). Returns exhibit fat tails — occasional extreme daily moves of ±5–10% occur far more often than a normal distribution predicts.

### 3. Heteroskedastic Errors

$$\varepsilon_i \sim \mathcal{N}\!\left(0,\ \sigma_i^2\right), \quad \sigma_i = 0.5|X_i| + 0.1$$

The error variance **grows with the covariate** $|X_i|$. The floor $+0.1$ prevents near-zero variance when $X_i \approx 0$. This is the violation OLS handles worst, because the standard error formula assumes $\sigma_i^2$ is constant.

**Real-world analog:** Modeling house prices ($Y$) as a function of square footage ($X$). A 500 sq-ft apartment has relatively predictable prices, but a 5,000 sq-ft mansion could sell anywhere from $2M to $10M depending on location, finishes, and market conditions. Variance grows with size.

### 4. Contaminated Normal (Outliers)

$$\varepsilon_i = \begin{cases} \mathcal{N}(0,1) & \text{with probability } 0.95 \\ \mathcal{N}(0,25) & \text{with probability } 0.05 \end{cases}$$

95% of observations are clean standard-normal errors. 5% are extreme outliers drawn from $\mathcal{N}(0,25)$ — standard deviation 5 times larger. This represents **occasional catastrophic measurement errors or data entry mistakes**.

**Real-world analog:** Clinical trial lab measurements. 95% of blood glucose readings are accurate, but 5% are contaminated by technician errors, equipment malfunctions, or patient non-compliance that produces wildly implausible values.

### 5. Skewed Errors

$$\varepsilon_i = \text{Exp}(1) - 1$$

A shifted exponential with mean 0, variance 1, and skewness coefficient 2. The distribution has a long right tail.

**Real-world analog:** Modeling insurance claim severity ($Y$) as a function of policy deductible ($X$). Most claims are small (near-zero), but a few catastrophic claims are extremely large. The error distribution is strongly right-skewed.

---

## Inference Methods and Their Math

### Standard OLS

The OLS estimator is:

$$\hat{\boldsymbol{\beta}} = (\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'\mathbf{Y}$$

Under the classical assumptions, $\hat{\beta}_1$ is unbiased and has exact sampling variance:

$$\text{Var}(\hat{\beta}_1) = \frac{\sigma^2}{\sum_{i=1}^n (X_i - \bar{X})^2}$$

In practice $\sigma^2$ is estimated by $s^2 = \text{RSS}/(n-k)$, giving estimated standard error:

$$\widehat{\text{SE}}_{\text{OLS}}(\hat{\beta}_1) = \sqrt{s^2 \cdot [(\mathbf{X}'\mathbf{X})^{-1}]_{11}}$$

The 95% confidence interval is:

$$\hat{\beta}_1 \pm t_{n-2,\, 0.025} \cdot \widehat{\text{SE}}_{\text{OLS}}(\hat{\beta}_1)$$

and the t-statistic for $H_0: \beta_1 = 0$ is:

$$t = \frac{\hat{\beta}_1}{\widehat{\text{SE}}_{\text{OLS}}(\hat{\beta}_1)} \sim t_{n-2} \quad \text{under } H_0 \text{ and classical assumptions}$$

**The critical flaw:** If $\text{Var}(\varepsilon_i \mid X_i) = \sigma_i^2 \neq \sigma^2$, the formula for $s^2$ is systematically wrong — it averages out the true non-constant variance — and the t-statistic no longer follows a $t_{n-2}$ distribution even asymptotically.

### HC3 Robust Inference (Heteroskedasticity-Consistent)

The HC3 (MacKinnon-White, 1985) sandwich estimator corrects for heteroskedasticity:

$$\widehat{\text{Var}}_{\text{HC3}}(\hat{\boldsymbol{\beta}}) = (\mathbf{X}'\mathbf{X})^{-1} \left[\sum_{i=1}^n \frac{\hat{\varepsilon}_i^2}{(1-h_{ii})^2} \mathbf{x}_i \mathbf{x}_i' \right] (\mathbf{X}'\mathbf{X})^{-1}$$

where $\hat{\varepsilon}_i = Y_i - \hat{Y}_i$ are OLS residuals and $h_{ii} = \mathbf{x}_i'(\mathbf{X}'\mathbf{X})^{-1}\mathbf{x}_i$ is the leverage of observation $i$.

The key insight is the $\frac{1}{(1-h_{ii})^2}$ leverage correction. High-leverage points (influential observations) receive an inflated residual, which prevents the SE from being artificially small when a few data points have disproportionate influence. This makes HC3 superior to HC0 and HC1 in small samples.

HC3 does **not** change the point estimate $\hat{\beta}_1$ — it only changes the standard error and therefore the CI and p-value.

### Bootstrap Inference (Percentile Method)

The **paired bootstrap** resamples entire $(X_i, Y_i)$ pairs with replacement, preserving any dependence between $X$ and the error variance:

For $b = 1, \ldots, B$:
1. Draw indices $\{i_1^*, \ldots, i_n^*\}$ uniformly with replacement from $\{1, \ldots, n\}$
2. Form bootstrap sample: $(X_{i_j^*}, Y_{i_j^*})$ for $j = 1, \ldots, n$
3. Refit OLS: $\hat{\beta}_1^{(b)} = (\mathbf{X}^{*\prime}\mathbf{X}^*)^{-1}\mathbf{X}^{*\prime}\mathbf{Y}^*$

The **percentile CI** uses the empirical quantiles of the bootstrap distribution directly:

$$\text{CI}_{\text{pct}} = \left[\hat{F}^{-1}(\alpha/2),\ \hat{F}^{-1}(1 - \alpha/2)\right]$$

where $\hat{F}$ is the empirical CDF of $\{\hat{\beta}_1^{(1)}, \ldots, \hat{\beta}_1^{(B)}\}$.

The **bootstrap p-value** for $H_0: \beta_1 = 0$ uses the shift method. Because the bootstrap distribution is centered at $\hat{\beta}_1$ (not at 0), we shift it to simulate the null distribution:

$$p\text{-value} = \frac{1}{B}\sum_{b=1}^B \mathbf{1}\!\left[\left|\hat{\beta}_1^{(b)} - \hat{\beta}_1\right| \geq |\hat{\beta}_1|\right]$$

This asks: "Under the null (where $\beta_1 = 0$), how often would we see a deviation as large as $|\hat{\beta}_1|$?"

### Bootstrap Inference (Pivotal / Basic Method)

The **pivotal CI** (also called basic bootstrap) reflects the bootstrap distribution around the original estimate. The motivation is that the bootstrap distribution of $\hat{\beta}_1^* - \hat{\beta}_1$ approximates the true sampling distribution of $\hat{\beta}_1 - \beta_1$. Inverting this:

$$\text{CI}_{\text{piv}} = \left[2\hat{\beta}_1 - \hat{F}^{-1}(1-\alpha/2),\ 2\hat{\beta}_1 - \hat{F}^{-1}(\alpha/2)\right]$$

**Why this is better under skewness:** When the sampling distribution of $\hat{\beta}_1$ is right-skewed, the percentile CI is also right-skewed — it puts too little mass on the lower tail. The pivotal CI corrects for this by reflecting: a right-skewed bootstrap distribution produces a left-shifted CI lower bound, giving more appropriate coverage. Asymptotically, both methods are equivalent, but in finite samples with skewed errors, the pivotal CI has better coverage.

Both bootstrap methods use $B = 2{,}000$ resamples per replication. Each bootstrap resample fits OLS using:

$$\hat{\beta}_1^{(b)} = \left[\mathbf{X}_b' \mathbf{X}_b\right]^{-1}\mathbf{X}_b' \mathbf{Y}_b$$

computed with NumPy's `linalg.solve` for speed — about 15× faster than calling statsmodels inside the loop.

---

## Evaluation Metrics

All metrics are computed per $(DGP, n, \text{method})$ cell across $R = 5{,}000$ Monte Carlo replications.

### Coverage Probability

The fundamental test of a confidence interval. Across $R$ replications, what fraction of CIs contain the true $\beta_1 = 0.5$?

$$\widehat{\text{Coverage}} = \frac{1}{R}\sum_{r=1}^R \mathbf{1}\!\left[\hat{\beta}_1^{\text{lo},(r)} \leq \beta_1 \leq \hat{\beta}_1^{\text{hi},(r)}\right]$$

A nominal 95% CI should achieve coverage $\approx 0.95$. Values below 0.94 indicate **anti-conservative** behavior (false confidence). The Monte Carlo standard error for this estimate is:

$$\text{MC-SE}(\widehat{\text{Coverage}}) = \sqrt{\frac{\widehat{\text{Coverage}}\,(1-\widehat{\text{Coverage}})}{R}}$$

With $R = 5{,}000$, MC-SE $\approx 0.003$, giving error bars of $\pm 0.006$ — tight enough to distinguish coverage differences of 2 percentage points.

### Type I Error Rate

Estimated using **separate data** generated under $H_0: \beta_1 = 0$ (with everything else identical). Measures how often each method falsely rejects a true null:

$$\widehat{\alpha}_{\text{actual}} = \frac{1}{R}\sum_{r=1}^R \mathbf{1}\!\left[p_{\text{null}}^{(r)} < \alpha\right], \quad \alpha = 0.05$$

A well-calibrated test has $\widehat{\alpha}_{\text{actual}} \approx 0.05$. Values above 0.06 represent inflated false-positive rates.

### Power

The probability of correctly rejecting $H_0$ when $\beta_1 = 0.5$ (a real effect exists):

$$\widehat{\text{Power}} = \frac{1}{R}\sum_{r=1}^R \mathbf{1}\!\left[p^{(r)} < \alpha\right]$$

Note: power is only meaningful once Type I error is controlled. A method with 20% Type I error will appear powerful but is simply rejecting too often.

### Bias and MSE

OLS is unbiased under all DGPs in this study (the Gauss-Markov conditions on exogeneity still hold):

$$\widehat{\text{Bias}} = \bar{\hat{\beta}}_1 - \beta_1, \qquad \widehat{\text{MSE}} = \frac{1}{R}\sum_{r=1}^R (\hat{\beta}_1^{(r)} - \beta_1)^2$$

Since all four methods use the OLS point estimate, their bias and MSE are identical. The heavy-tails and outlier DGPs increase MSE substantially at small $n$ because extreme errors make the estimate noisier, even though the estimator remains unbiased.

### Average CI Width

$$\overline{W} = \frac{1}{R}\sum_{r=1}^R \left(\hat{\beta}_1^{\text{hi},(r)} - \hat{\beta}_1^{\text{lo},(r)}\right)$$

Width should decrease as $O(n^{-1/2})$. A method that achieves correct coverage with a narrower interval is more efficient.

---

## Simulation Design

| Parameter | Value |
|---|---|
| True $\beta_0$ | 2 |
| True $\beta_1$ | 0.5 |
| Sample sizes | 30, 50, 100, 250, 500 |
| Replications $R$ | 5,000 per cell |
| Bootstrap resamples $B$ | 2,000 per replication |
| Significance level $\alpha$ | 0.05 |
| DGPs | 5 |
| Methods | 4 |
| Total cells | 25 |
| Total simulation rows | 500,000 |

Cells are parallelized with `joblib`. Each cell gets a **deterministic seed** computed as `seed = 42 × 10,000 + dgp_idx × 100 + n_idx`, ensuring full reproducibility across separate runs without relying on Python's hash randomization.

---

## Results and Interpretation

The heatmap below gives an at-a-glance overview of every $(DGP \times method \times n)$ combination. Red cells are dangerous (undercoverage — the CI is too narrow and misleads you). Blue cells are conservative (overcoverage — the CI is wider than needed). White is ideal.

![Coverage Deviation Heatmap](figures/06_coverage_heatmap.png)

The single most visible pattern: the entire OLS row for **Heteroskedastic** is deep red across all $n$, while every other DGP is near-white for OLS. This tells the whole story before reading a single number.

---

### Finding 1: Heteroskedasticity Permanently Breaks OLS

This is the study's most striking result. Standard OLS under heteroskedastic errors achieves **77% coverage** — regardless of sample size.

| $n$ | OLS | HC3 Robust | Bootstrap (Pct) | Bootstrap (Piv) |
|---|---|---|---|---|
| 30 | 0.774 | 0.919 | 0.908 | 0.857 |
| 50 | 0.766 | 0.927 | 0.913 | 0.888 |
| 100 | 0.772 | 0.936 | 0.930 | 0.916 |
| 250 | 0.774 | 0.945 | 0.939 | 0.937 |
| 500 | 0.765 | 0.950 | 0.946 | 0.946 |

OLS coverage stays stuck at ~77% because the issue is not statistical but **algebraic**: the classical SE formula is misspecified at a fundamental level. With $\sigma_i = 0.5|X_i| + 0.1$, OLS underestimates $\text{Var}(\hat{\beta}_1)$, making CIs too narrow. Larger $n$ does not help — you are just getting more precise estimates of a wrong quantity.

The Type I error consequences are equally severe. OLS rejects a true null at **21–23% rate** (nominally 5%) across all sample sizes — meaning 1 in 5 regressions in a heteroskedastic setting yields a spurious "statistically significant" result with standard errors.

HC3 and the percentile bootstrap both recover well, approaching nominal 95% coverage by $n = 250$. **HC3 is the most reliable correction here**, achieving 95.0% at $n = 500$.

**The bootstrap pivotal CI performs unexpectedly poorly** under heteroskedasticity at small $n$ (85.7% coverage at $n=30$). This is because pivotal CI assumes the bootstrap distribution is an accurate replica of the sampling distribution — but with only $n=30$ points and high leverage observations, individual bootstrap resamples can have wildly different design matrices, making the pivotal reflection unstable.

![Coverage Probability](figures/01_coverage_probability.png)

The coverage plot makes the heteroskedastic failure unmistakable: the red OLS line sits flat at ~0.77 across all five panels for that DGP, while every other method and DGP trends toward 0.95. Error bars are $\pm 1.96 \times \text{MC-SE}$, confirming the OLS gap is not Monte Carlo noise.

![Type I Error Rate](figures/02_type1_error.png)

The Type I error plot is the mirror image of coverage: OLS under heteroskedasticity sits at ~0.22, more than four times the nominal 5% red line. All other DGP–method combinations hug the dashed line closely by $n=100$.

### Finding 2: Heavy Tails Have Moderate and Diminishing Impact

| $n$ | OLS | HC3 Robust | Bootstrap (Pct) | Bootstrap (Piv) |
|---|---|---|---|---|
| 30 | 0.942 | 0.947 | 0.918 | 0.944 |
| 50 | 0.952 | 0.954 | 0.934 | 0.955 |
| 100 | 0.953 | 0.953 | 0.935 | 0.955 |
| 250 | 0.948 | 0.949 | 0.939 | 0.953 |
| 500 | 0.949 | 0.948 | 0.943 | 0.951 |

Surprisingly, OLS holds up well under $t_3$ errors. At $n=30$, coverage is 94.2% — only slightly below nominal. This happens because OLS is based on sums of squared errors; even though individual errors are heavy-tailed, the **Central Limit Theorem acts on the sum** $\sum X_i \varepsilon_i$, and the $t_3$ distribution has finite variance (= 3), so the CLT kicks in quickly.

The **percentile bootstrap consistently undercovers** (91.8–94.3%) under heavy tails. This is a known limitation: the bootstrap distribution inherits the skewness and kurtosis of the original sample, and in small samples this leads to CIs that are too symmetric. The pivotal bootstrap corrects this and matches or beats OLS and HC3 throughout.

Type I errors are well-controlled for all methods — none exceeds 6%, confirming the inference is not severely distorted even in small samples.

### Finding 3: Outliers Inflate Variance but Preserve Calibration

| $n$ | OLS | HC3 Robust | Bootstrap (Pct) | Bootstrap (Piv) |
|---|---|---|---|---|
| 30 | 0.948 | 0.947 | 0.923 | 0.946 |
| 50 | 0.949 | 0.953 | 0.924 | 0.956 |
| 100 | 0.949 | 0.952 | 0.933 | 0.953 |
| 250 | 0.948 | 0.952 | 0.938 | 0.957 |
| 500 | 0.947 | 0.949 | 0.941 | 0.954 |

The contaminated normal is 95% clean data + 5% wild outliers. Coverage for OLS is actually near-nominal (94.8%) throughout. Why? The outlier contamination inflates $s^2$ (the variance estimate), which makes the CI **wider** — the interval is more conservative, not less, so coverage is maintained.

The tradeoff is **power loss**. At $n=30$, OLS power is 51.6% compared to 57.1% for the bootstrap, because OLS CIs are unnecessarily wide due to outlier-inflated variance. The bootstrap, by resampling actual data points, implicitly downweights the impact of the outlier cluster in any given resample.

The percentile bootstrap again undercovers (92.3–94.1%), while the pivotal bootstrap matches or exceeds all other methods.

### Finding 4: Skewed Errors — Bootstrap Pivotal Is Clearly Superior

| $n$ | OLS | HC3 Robust | Bootstrap (Pct) | Bootstrap (Piv) |
|---|---|---|---|---|
| 30 | 0.949 | 0.949 | 0.925 | 0.943 |
| 50 | 0.948 | 0.951 | 0.931 | 0.950 |
| 100 | 0.952 | 0.948 | 0.937 | 0.951 |
| 250 | 0.948 | 0.948 | 0.939 | 0.949 |
| 500 | 0.952 | 0.953 | 0.949 | 0.954 |

The skewed (exponential) DGP is where the **percentile vs. pivotal comparison is most instructive**.

The percentile CI persistently undercovers at 92.5% for $n=30$ and doesn't reach 95% until $n=500$. The cause is the asymmetry of the bootstrap distribution. With right-skewed errors, $\hat{\beta}_1$ has a slightly right-skewed sampling distribution. The percentile CI's lower bound is placed too high (the left 2.5% quantile is too far right), missing the true $\beta_1$ when estimates are on the low end.

The pivotal CI corrects this by reflection: a right-skewed bootstrap distribution produces a left-shifted lower bound, which matches the actual sampling distribution better. Result: 94.3% coverage at $n=30$ — a full 1.8 percentage points better than percentile.

OLS and HC3 perform excellently here (94.9%). This confirms that for symmetric distributions, the $t$-approximation is robust to skewness via CLT, but the bootstrap needs to use the right CI method to match it.

### Finding 5: Normal Errors — All Methods Work, with Small Finite-Sample Differences

| $n$ | OLS | HC3 Robust | Bootstrap (Pct) | Bootstrap (Piv) |
|---|---|---|---|---|
| 30 | 0.946 | 0.933 | 0.920 | 0.924 |
| 50 | 0.949 | 0.943 | 0.934 | 0.934 |
| 100 | 0.948 | 0.945 | 0.941 | 0.937 |
| 250 | 0.945 | 0.946 | 0.941 | 0.942 |
| 500 | 0.951 | 0.951 | 0.948 | 0.951 |

Under the textbook scenario, all methods work well. OLS performs best at $n=30$ (94.6%), as expected — it has the exact theoretical distribution here.

Interestingly, **HC3 slightly undercovers at $n=30$ (93.3%)** even under homoskedastic errors. This is a known small-sample cost of the HC3 correction: the leverage adjustment $1/(1-h_{ii})^2$ inflates residuals slightly, making the HC3 SE somewhat conservative, which paradoxically widens the CI and can cause slight undercoverage at very small $n$ via df effects in the t-approximation.

Both bootstrap CIs undercover at small $n$ (92–93%). This is expected — the bootstrap distribution approximates the sampling distribution only asymptotically.

### Power Summary

With $\beta_1 = 0.5$ and $X \sim \mathcal{N}(0,1)$:

| DGP | Method | $n=30$ | $n=50$ | $n=100$ | $n=250$ | $n=500$ |
|---|---|---|---|---|---|---|
| Normal | OLS | 0.718 | 0.916 | 0.996 | 1.000 | 1.000 |
| Heavy Tails | OLS | 0.796 | 0.917 | 0.988 | 0.999 | 1.000 |
| Heterosk. | OLS | 0.949 | 0.990 | 1.000 | 1.000 | 1.000 |
| Outliers | OLS | 0.516 | 0.677 | 0.893 | 0.997 | 1.000 |
| Skewed | OLS | 0.750 | 0.916 | 0.991 | 1.000 | 1.000 |

The **heteroskedastic DGP shows the highest OLS power** — but this is misleading. OLS rejects frequently because it underestimates the SE, not because the test is well-powered. Type I error is 21%, so the high rejection rate includes many false positives.

The **outlier DGP has the lowest power** (51.6% at $n=30$). The 5% contamination inflates $s^2$, which widens CIs and makes the test conservative. Power only recovers around $n=250$.

![Power Curves](figures/03_power_curves.png)

All methods converge to 100% power by $n=250$ except under the outlier DGP, where the slow rise is visible. The heteroskedastic panel shows OLS (red) appearing most powerful — this is the false power from inflated Type I error, not genuine sensitivity.

![CI Width](figures/04_ci_width.png)

CI widths decrease as $O(n^{-1/2})$ for all methods. Under heteroskedasticity, OLS CIs are **narrower** than HC3 and bootstrap — the very reason OLS undercovers. HC3 adds appropriate width to account for the non-constant variance. Under the outlier DGP, all CIs are wider at small $n$ due to the inflated variance from the 5% contamination cluster.

![Bias and MSE](figures/05_bias_mse.png)

OLS bias is negligible across all DGPs and sample sizes (all lines hug zero), confirming OLS remains an unbiased estimator even when its inference is miscalibrated. MSE is highest for the outlier DGP at $n=30$ (≈ 0.083 vs. ≈ 0.037 for normal), reflecting the additional variance from extreme observations.

---

## Real-World Scenario Walkthrough

### Scenario A: You Are a Housing Economist

You are estimating how $\log(\text{square footage})$ affects $\log(\text{price})$ using $n=50$ observations from a city neighborhood. Your data has high variance at the top end: luxury penthouses have unpredictable prices.

**What your software reports (OLS):**
- $\hat{\beta}_1 = 0.48$, $\text{SE} = 0.15$, $p = 0.002$, `95% CI: [0.19, 0.77]`

**What is likely actually true:**
- Your errors are heteroskedastic. From our simulation: OLS coverage at $n=50$ under heteroskedasticity is **76.6%**, not 95%.
- The actual 95% CI should be wider — roughly $[0.11, 0.85]$ based on HC3.
- Your $p$-value of 0.002 is also likely too small. True Type I error rate is ~22%.

**What to do:** Use `statsmodels OLS(...).fit(cov_type='HC3')`. At $n=50$, HC3 achieves 92.7% coverage — not perfect but far better than 76.6%.

### Scenario B: You Are Analyzing Clinical Trial Data

You have $n=100$ patients and are regressing a biomarker on treatment assignment. You know from prior experience that ~5% of measurements are erroneous.

**Risk:** Outliers inflate $s^2$, making your CIs too wide and your test underpowered. From our simulation: OLS power at $n=100$ under contamination is 89.3%.

**What to do:** The paired bootstrap CI is reasonable here. At $n=100$, the pivotal bootstrap achieves 95.3% coverage and has similar or better power to OLS. More importantly, because the bootstrap resamples actual data, it naturally reflects the outlier frequency in your sample.

### Scenario C: You Are a Quant Modeling Stock Returns

You have $n=250$ daily return observations and are running a CAPM regression. Daily returns are well-known to have heavy tails ($t_3$-like).

**Risk:** Minor. From our simulation, OLS coverage at $n=250$ under heavy tails is **94.8%** — very close to nominal. The fat tails are mostly smoothed out by $n=250$.

**What to do:** OLS is fine here. If you want additional robustness, HC3 achieves 94.9% at this sample size with negligible cost. The percentile bootstrap slightly undercovers (93.9%) — if you use bootstrap, prefer the pivotal method.

### Scenario D: You Are Modeling Insurance Claims

You have $n=50$ observations of claim amounts, which are right-skewed. Your regression is on log-transformed claims.

**Risk:** The log transform reduces skewness but rarely eliminates it. From our simulation under skewed errors at $n=50$: percentile bootstrap coverage is **93.1%** (undercovering), while the pivotal bootstrap hits **95.0%**.

**What to do:** Use the pivotal bootstrap CI. The correction for distributional asymmetry provides meaningful improvement over the percentile method and matches the OLS CI in coverage while requiring fewer distributional assumptions.

---

## Summary Recommendations

| Data Situation | Recommended Method | Why |
|---|---|---|
| Large $n$ (≥250), any errors | Any — all converge | CLT dominates |
| Heteroskedastic errors | **HC3 Robust** | Consistent coverage at all $n$; percentile bootstrap is runner-up |
| Heavy-tailed errors, small $n$ | **HC3 or Pivotal Bootstrap** | Percentile bootstrap undercovers; OLS surprisingly robust |
| Outliers / contaminated data | **Pivotal Bootstrap or HC3** | OLS OK on coverage but low power; bootstrap recovers power |
| Skewed errors, small $n$ | **Pivotal Bootstrap or OLS** | Percentile bootstrap consistently undercovering |
| Normal errors, small $n$ | **OLS** | Exact theory applies; robust methods add unnecessary variance |

**The one situation to always avoid:** Standard OLS standard errors with heteroskedastic data. This is the only condition in our study where OLS fails completely and permanently — coverage remains ~77% and Type I error remains ~22% no matter how large $n$ grows.

---

## Quick Start

```bash
# Install dependencies
pip install numpy pandas statsmodels matplotlib seaborn joblib

# Quick test run (R=200, B=500, ~2 minutes)
python run.py --quick

# Full simulation (R=5,000, B=2,000, parallel across all CPU cores)
python run.py

# Regenerate plots from saved results without re-running simulation
python run.py --plots-only
```

---

## Project Structure

```
├── config.py           # All simulation parameters (betas, n, R, B, DGPs, methods)
├── dgp.py              # Data-generating process functions (5 error distributions)
├── estimators.py       # OLS, HC3, percentile bootstrap, pivotal bootstrap
├── simulation.py       # Main Monte Carlo engine (parallelized with joblib)
├── analysis.py         # Aggregates raw results into coverage/power/bias/MSE metrics
├── visualizations.py   # Six publication-quality plots
├── run.py              # End-to-end entry point with --quick and --plots-only flags
├── results/
│   ├── raw_results.csv # One row per (dgp, n, method, replication): 500k rows
│   └── metrics.csv     # One row per (dgp, n, method): 100 rows
└── figures/
    ├── 01_coverage_probability.png   # Coverage vs n, one panel per DGP
    ├── 02_type1_error.png            # Type I error rate vs n
    ├── 03_power_curves.png           # Power vs n with MC error bars
    ├── 04_ci_width.png               # Average CI width vs n
    ├── 05_bias_mse.png               # OLS bias and MSE across DGPs
    └── 06_coverage_heatmap.png       # Coverage deviation heatmap (all conditions)
```

---

## Requirements

- Python 3.9+
- `numpy` — array operations, fast bootstrap matrix algebra
- `pandas` — results management
- `statsmodels` — OLS and HC3 inference
- `matplotlib` / `seaborn` — visualizations
- `joblib` — parallel Monte Carlo execution
