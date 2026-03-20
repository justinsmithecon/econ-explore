"""Reusable matplotlib helpers for distribution plots and shading."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


def plot_analytic_distributions(ax, info: dict, power: float):
    """Plot analytic null and true distributions with rejection region and power shading."""
    null_dist = info["null_dist"]
    true_dist = info["true_dist"]
    alpha = info["alpha"]

    crit_lower = null_dist.ppf(alpha / 2)
    crit_upper = null_dist.ppf(1 - alpha / 2)

    x_lo = min(null_dist.ppf(0.001), true_dist.ppf(0.001))
    x_hi = max(null_dist.ppf(0.999), true_dist.ppf(0.999))
    x = np.linspace(x_lo, x_hi, 500)

    null_pdf = null_dist.pdf(x)
    true_pdf = true_dist.pdf(x)

    ax.plot(x, null_pdf, color="#2563eb", linewidth=2, label="Null (H₀)")
    ax.plot(x, true_pdf, color="#dc2626", linewidth=2, label="True (H₁)")

    mask_reject_lower = x <= crit_lower
    mask_reject_upper = x >= crit_upper
    ax.fill_between(x, null_pdf, where=mask_reject_lower,
                    color="#2563eb", alpha=0.15)
    ax.fill_between(x, null_pdf, where=mask_reject_upper,
                    color="#2563eb", alpha=0.15, label=f"Rejection region (α={alpha})")
    ax.fill_between(x, true_pdf, where=mask_reject_lower,
                    color="#dc2626", alpha=0.25)
    ax.fill_between(x, true_pdf, where=mask_reject_upper,
                    color="#dc2626", alpha=0.25, label=f"Power = {power:.3f}")

    ax.axvline(x=crit_lower, color="#64748b", linestyle="--", linewidth=1, alpha=0.7)
    ax.axvline(x=crit_upper, color="#64748b", linestyle="--", linewidth=1, alpha=0.7)

    ax.set_xlabel(info["stat_name"], fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.2)
    ax.set_ylim(bottom=0)


def plot_simulated_distributions(ax, sim_values: np.ndarray, info: dict, alpha: float, power: float):
    """Plot histogram of simulated values with analytic curves overlaid."""
    null_dist = info["null_dist"]
    true_dist = info["true_dist"]

    crit_lower = null_dist.ppf(alpha / 2)
    crit_upper = null_dist.ppf(1 - alpha / 2)

    bin_edges = np.histogram_bin_edges(sim_values, bins=60)
    counts, _ = np.histogram(sim_values, bins=bin_edges)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_widths = np.diff(bin_edges)
    density = counts / (counts.sum() * bin_widths)

    colors = ["#dc2626" if c <= crit_lower or c >= crit_upper else "#f4a6a6"
              for c in bin_centers]
    ax.bar(bin_centers, density, width=bin_widths, color=colors,
           edgecolor="white", linewidth=0.5)

    x_lo = min(null_dist.ppf(0.001), true_dist.ppf(0.001), sim_values.min())
    x_hi = max(null_dist.ppf(0.999), true_dist.ppf(0.999), sim_values.max())
    x = np.linspace(x_lo, x_hi, 500)
    null_line, = ax.plot(x, null_dist.pdf(x), color="#2563eb", linewidth=1.5,
                         linestyle="--", alpha=0.7)
    true_line, = ax.plot(x, true_dist.pdf(x), color="#64748b", linewidth=1.5,
                         linestyle="--", alpha=0.7)

    crit_line = ax.axvline(x=crit_lower, color="#1e3a5f", linestyle="--",
                           linewidth=1.2, alpha=0.8)
    ax.axvline(x=crit_upper, color="#1e3a5f", linestyle="--", linewidth=1.2, alpha=0.8)

    ax.legend(handles=[
        Patch(facecolor="#f4a6a6", edgecolor="white", label="Not rejected"),
        Patch(facecolor="#dc2626", edgecolor="white", label=f"Rejected (power = {power:.3f})"),
        null_line, true_line, crit_line,
    ], labels=[
        "Not rejected",
        f"Rejected (power = {power:.3f})",
        "Null (H₀) theory",
        "True (H₁) theory",
        f"Critical values (α={alpha})",
    ], loc="best", fontsize=8)

    ax.set_xlabel(info["stat_name"], fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.set_ylim(bottom=0)


def plot_histogram(ax, values: np.ndarray, *, bins: int = 50,
                   color: str = "#2563eb", label: str = "", alpha: float = 0.7):
    """Simple density histogram helper."""
    ax.hist(values, bins=bins, density=True, color=color, alpha=alpha,
            edgecolor="white", linewidth=0.5, label=label)
    ax.set_ylabel("Density", fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.set_ylim(bottom=0)
