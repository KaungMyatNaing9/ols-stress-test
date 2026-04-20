"""
Visualizations for Simulation Study
Generates publication-quality plots for presentation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import CONFIG

sns.set_theme(style="whitegrid", font_scale=1.1)

METHOD_COLORS = {
    "OLS": "#e74c3c",
    "HC3_Robust": "#2ecc71",
    "Bootstrap_Percentile": "#3498db",
    "Bootstrap_Pivotal": "#9b59b6",
}
METHOD_MARKERS = {
    "OLS": "o",
    "HC3_Robust": "s",
    "Bootstrap_Percentile": "D",
    "Bootstrap_Pivotal": "^",
}


def _method_label(method):
    return CONFIG["method_labels"].get(method, method)


def plot_coverage(metrics, save_path="figures/01_coverage_probability.png"):
    """
    Coverage probability vs sample size, one panel per DGP.
    y-axis adapts to data so undercoverage is never silently clipped.
    """
    dgp_names = CONFIG["dgp_names"]
    dgp_labels = CONFIG["dgp_labels"]
    nominal = 1 - CONFIG["alpha"]

    # Dynamic y-axis: floor at min observed coverage minus margin, never above 1
    y_min = max(0.65, metrics["coverage"].min() - 0.04)
    y_max = 1.01

    fig, axes = plt.subplots(1, len(dgp_names), figsize=(4 * len(dgp_names), 4.5),
                             sharey=True)
    if len(dgp_names) == 1:
        axes = [axes]

    for ax, dgp in zip(axes, dgp_names):
        dgp_data = metrics[metrics["dgp"] == dgp]

        for method in CONFIG["method_names"]:
            method_data = dgp_data[dgp_data["method"] == method].sort_values("n")
            ax.errorbar(
                method_data["n"], method_data["coverage"],
                yerr=1.96 * method_data["mc_se_coverage"],
                label=_method_label(method),
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                markersize=6, linewidth=2, capsize=3,
            )

        ax.axhline(y=nominal, color="black", linestyle="--",
                   linewidth=1, alpha=0.7, label=f"Nominal ({nominal:.2f})")
        ax.axhspan(nominal - 0.01, nominal + 0.01, alpha=0.08, color="green")
        ax.set_title(dgp_labels.get(dgp, dgp), fontsize=11, fontweight="bold")
        ax.set_xlabel("Sample Size (n)")
        ax.set_xticks(CONFIG["sample_sizes"])
        ax.set_ylim(y_min, y_max)

    axes[0].set_ylabel("Coverage Probability")
    axes[0].legend(loc="lower right", fontsize=8)

    fig.suptitle("Coverage Probability of 95% Confidence Intervals",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_type1_error(metrics, save_path="figures/02_type1_error.png"):
    """
    Type I Error Rate: how often each method falsely rejects H0 when H0 is true.
    """
    dgp_names = CONFIG["dgp_names"]
    dgp_labels = CONFIG["dgp_labels"]

    fig, axes = plt.subplots(1, len(dgp_names), figsize=(4 * len(dgp_names), 4.5),
                             sharey=True)
    if len(dgp_names) == 1:
        axes = [axes]

    for ax, dgp in zip(axes, dgp_names):
        dgp_data = metrics[metrics["dgp"] == dgp]

        for method in CONFIG["method_names"]:
            method_data = dgp_data[dgp_data["method"] == method].sort_values("n")
            ax.errorbar(
                method_data["n"], method_data["type1_error"],
                yerr=1.96 * method_data["mc_se_type1"],
                label=_method_label(method),
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                markersize=6, linewidth=2, capsize=3,
            )

        ax.axhline(y=CONFIG["alpha"], color="black", linestyle="--",
                   linewidth=1, alpha=0.7, label=f"Nominal ({CONFIG['alpha']})")
        ax.axhspan(0.04, 0.06, alpha=0.08, color="green")
        ax.set_title(dgp_labels.get(dgp, dgp), fontsize=11, fontweight="bold")
        ax.set_xlabel("Sample Size (n)")
        ax.set_xticks(CONFIG["sample_sizes"])

    axes[0].set_ylabel("Type I Error Rate")
    axes[0].set_ylim(0.0, 0.15)
    axes[0].legend(loc="upper right", fontsize=8)

    fig.suptitle("Type I Error Rate (Nominal = 0.05)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_power(metrics, save_path="figures/03_power_curves.png"):
    """
    Power: probability of correctly rejecting H0 when true beta_1 ≠ 0.
    Error bars show ±1.96 × Monte Carlo SE for honest uncertainty quantification.
    """
    dgp_names = CONFIG["dgp_names"]
    dgp_labels = CONFIG["dgp_labels"]

    fig, axes = plt.subplots(1, len(dgp_names), figsize=(4 * len(dgp_names), 4.5),
                             sharey=True)
    if len(dgp_names) == 1:
        axes = [axes]

    for ax, dgp in zip(axes, dgp_names):
        dgp_data = metrics[metrics["dgp"] == dgp]

        for method in CONFIG["method_names"]:
            method_data = dgp_data[dgp_data["method"] == method].sort_values("n")
            ax.errorbar(
                method_data["n"], method_data["power"],
                yerr=1.96 * method_data["mc_se_power"],
                label=_method_label(method),
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                markersize=6, linewidth=2, capsize=3,
            )

        ax.set_title(dgp_labels.get(dgp, dgp), fontsize=11, fontweight="bold")
        ax.set_xlabel("Sample Size (n)")
        ax.set_xticks(CONFIG["sample_sizes"])
        ax.set_ylim(0, 1.05)

    axes[0].set_ylabel("Power")
    axes[0].legend(loc="lower right", fontsize=8)

    beta1 = CONFIG["beta_1"]
    fig.suptitle(f"Power (Correctly Rejecting H₀: β₁ = 0 when true β₁ = {beta1})",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_ci_width(metrics, save_path="figures/04_ci_width.png"):
    """Average CI Width: narrower is better, but only meaningful when coverage holds."""
    dgp_names = CONFIG["dgp_names"]
    dgp_labels = CONFIG["dgp_labels"]

    fig, axes = plt.subplots(1, len(dgp_names), figsize=(4 * len(dgp_names), 4.5),
                             sharey=False)
    if len(dgp_names) == 1:
        axes = [axes]

    for ax, dgp in zip(axes, dgp_names):
        dgp_data = metrics[metrics["dgp"] == dgp]

        for method in CONFIG["method_names"]:
            method_data = dgp_data[dgp_data["method"] == method].sort_values("n")
            ax.plot(
                method_data["n"], method_data["avg_ci_width"],
                label=_method_label(method),
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                markersize=6, linewidth=2,
            )

        ax.set_title(dgp_labels.get(dgp, dgp), fontsize=11, fontweight="bold")
        ax.set_xlabel("Sample Size (n)")
        ax.set_xticks(CONFIG["sample_sizes"])

    axes[0].set_ylabel("Average CI Width")
    axes[0].legend(loc="upper right", fontsize=8)

    fig.suptitle("Average Confidence Interval Width",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_bias_mse(metrics, save_path="figures/05_bias_mse.png"):
    """
    Bias and MSE of the point estimator across DGPs and methods.

    All four methods use OLS as their point estimator, so bias and MSE
    are identical for OLS/HC3/Bootstrap variants. Plotting all methods
    confirms this and surfaces any small-sample bootstrap deviations.
    """
    dgp_names = CONFIG["dgp_names"]
    dgp_labels = CONFIG["dgp_labels"]
    true_beta = CONFIG["beta_1"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for panel, metric_col, ylabel, title in [
        (axes[0], "bias",  f"Bias (β̂₁ − {true_beta})", "Bias of Point Estimator"),
        (axes[1], "mse",   "Mean Squared Error",        "MSE of Point Estimator"),
    ]:
        # OLS and HC3 share the same estimator; show one line per DGP
        # Bootstrap variants may differ minutely in small samples
        for dgp in dgp_names:
            ols_data = metrics[
                (metrics["method"] == "OLS") & (metrics["dgp"] == dgp)
            ].sort_values("n")
            panel.plot(
                ols_data["n"], ols_data[metric_col],
                label=dgp_labels.get(dgp, dgp).replace("\n", " "),
                marker="o", markersize=5, linewidth=1.5,
            )

        if metric_col == "bias":
            panel.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.7)

        panel.set_xlabel("Sample Size (n)")
        panel.set_ylabel(ylabel)
        panel.set_title(title, fontweight="bold")
        panel.legend(fontsize=8)
        panel.set_xticks(CONFIG["sample_sizes"])

    fig.suptitle(
        "Point Estimation Performance (all methods share the OLS estimator)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_heatmap(metrics, save_path="figures/06_coverage_heatmap.png"):
    """
    Coverage deviation from nominal across all conditions.
    Red = undercoverage (anti-conservative), Blue = overcoverage (conservative).
    Asymmetric color scale: undercoverage is more dangerous so gets more visual range.
    """
    metrics = metrics.copy()
    metrics["label"] = (
        metrics["dgp"] + " | "
        + metrics["method"].map(CONFIG["method_labels"]).fillna(metrics["method"])
    )
    metrics["deviation"] = metrics["coverage"] - (1 - CONFIG["alpha"])

    pivot = metrics.pivot_table(index="label", columns="n", values="deviation")

    order = [
        f"{dgp} | {CONFIG['method_labels'].get(method, method)}"
        for dgp in CONFIG["dgp_names"]
        for method in CONFIG["method_names"]
        if f"{dgp} | {CONFIG['method_labels'].get(method, method)}" in pivot.index
    ]
    pivot = pivot.loc[order]

    fig, ax = plt.subplots(figsize=(11, len(order) * 0.55 + 2))
    sns.heatmap(
        pivot, annot=True, fmt=".3f", center=0,
        cmap="RdBu", linewidths=0.5,
        vmin=-0.10, vmax=0.05,
        cbar_kws={"label": "Coverage − 0.95"},
        ax=ax,
    )
    ax.set_xlabel("Sample Size (n)", fontsize=12)
    ax.set_ylabel("")
    ax.set_title(
        "Coverage Deviation from Nominal 95%\n"
        "(Red = Under-coverage [anti-conservative], Blue = Over-coverage [conservative])",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def generate_all_plots(metrics):
    """Generate all visualization plots."""
    print("\nGenerating plots...")
    print("-" * 40)
    plot_coverage(metrics)
    plot_type1_error(metrics)
    plot_power(metrics)
    plot_ci_width(metrics)
    plot_bias_mse(metrics)
    plot_heatmap(metrics)
    print("-" * 40)
    print("All plots generated!")


if __name__ == "__main__":
    metrics = pd.read_csv("results/metrics.csv")
    generate_all_plots(metrics)
