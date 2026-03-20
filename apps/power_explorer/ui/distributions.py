"""Two-panel distribution visualization: original statistic and test statistic."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from shared.plotting import plot_analytic_distributions, plot_simulated_distributions


def render_distributions(scenario, params: dict, method: str, sim_result: dict | None = None):
    """Render two side-by-side distribution panels with formulas."""
    power = scenario.analytic_power(params)
    stat_info = scenario.distribution_info(params)
    test_info = scenario.test_statistic_info(params)
    alpha = params["alpha"]

    if method == "Simulation" and sim_result is not None:
        sim_power = sim_result["power"]
    else:
        sim_power = power

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Sampling Distribution of the Statistic**")
        st.latex(stat_info["stat_formula"])
    with col2:
        st.markdown("**Distribution of the Test Statistic\\***")
        st.latex(test_info["stat_formula"])

    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        if method == "Analytic":
            plot_analytic_distributions(ax1, stat_info, power)
        else:
            plot_simulated_distributions(ax1, sim_result["sample_stats"], stat_info, alpha, sim_power)
        fig1.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

    with col2:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        if method == "Analytic":
            plot_analytic_distributions(ax2, test_info, power)
        else:
            plot_simulated_distributions(ax2, sim_result["test_stats"], test_info, alpha, sim_power)
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)
