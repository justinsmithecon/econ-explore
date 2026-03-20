"""Power curve plot: power vs sample size."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


def render_power_curve(scenario, params: dict, method: str, n_simulations: int = 5000, seed: int | None = 42):
    """Render the power vs sample size plot."""
    has_n_per_group = "n_per_group" in params
    n_key = "n_per_group" if has_n_per_group else "n"
    current_n = int(params[n_key])

    n_min = 10
    n_max = max(300, current_n + 50)
    n_step = 10 if method == "Simulation" else 5

    params_for_curve = {k: v for k, v in params.items() if k != n_key}
    n_range = np.arange(n_min, n_max + 1, n_step)

    powers = []
    for n in n_range:
        p = {**params_for_curve, n_key: int(n)}
        if method == "Analytic":
            powers.append(scenario.analytic_power(p))
        else:
            sim = scenario.simulate_power(p, n_simulations, seed)
            powers.append(sim["power"])
    powers = np.array(powers)

    if method == "Analytic":
        current_power = scenario.analytic_power(params)
    else:
        current_power = scenario.simulate_power(params, n_simulations, seed)["power"]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    label = "Analytic power" if method == "Analytic" else "Simulated power"
    style = "-" if method == "Analytic" else "o-"
    ax.plot(n_range, powers, style, color="#2563eb",
            linewidth=2, markersize=4, label=label, zorder=3)
    ax.axhline(y=0.80, color="#94a3b8", linestyle="--", linewidth=1,
               label="80% power", zorder=1)
    ax.axvline(x=current_n, color="#a855f7", linestyle=":", linewidth=1.5,
               alpha=0.7, label=f"Current n={current_n}", zorder=2)
    ax.plot(current_n, current_power, "o", color="#a855f7", markersize=8, zorder=5)

    ax.set_xlabel("Sample Size" + (" per group" if has_n_per_group else ""),
                  fontsize=11)
    ax.set_ylabel("Power", fontsize=11)
    ax.set_title("Power Curve", fontsize=13, fontweight="bold")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)
