"""Statistical Power Explorer — Streamlit app."""

import streamlit as st

from apps.power_explorer.scenarios import ALL_SCENARIOS
from apps.power_explorer.ui.distributions import render_distributions
from apps.power_explorer.ui.power_curve import render_power_curve
from apps.power_explorer.ui.education import render_education


def _render_sidebar():
    """Render the sidebar and return (scenario, params, method, n_simulations, seed)."""
    st.sidebar.header("Statistical Power Explorer")

    scenario_names = [s.name for s in ALL_SCENARIOS]
    selected = st.sidebar.selectbox("Choose a test", scenario_names)
    scenario = ALL_SCENARIOS[scenario_names.index(selected)]

    st.sidebar.markdown("---")
    st.sidebar.subheader("Parameters")

    params = {}
    for slider in scenario.sliders():
        val = st.sidebar.slider(
            slider.label,
            min_value=slider.min_value,
            max_value=slider.max_value,
            value=slider.default,
            step=slider.step,
            help=slider.help_text,
            key=f"{scenario.name}_{slider.key}",
        )
        params[slider.key] = val

    st.sidebar.markdown("---")
    method = st.sidebar.radio(
        "Power method",
        ["Analytic", "Simulation"],
        help="Analytic uses exact formulas; Simulation estimates power via Monte Carlo",
    )

    n_simulations = 5000
    seed = 42
    if method == "Simulation":
        n_simulations = st.sidebar.slider(
            "Number of simulations",
            min_value=500,
            max_value=20000,
            value=5000,
            step=500,
            help="More simulations = more precise estimate, but slower",
        )
        use_seed = st.sidebar.checkbox("Fix random seed", value=True,
                                        help="Enable for reproducible results")
        seed = 42 if use_seed else None

    return scenario, params, method, n_simulations, seed


def main():
    """Main entry point for the Statistical Power Explorer app."""
    scenario, params, method, n_simulations, seed = _render_sidebar()

    st.title("Statistical Power Explorer")
    st.markdown(f"### {scenario.name}")
    st.markdown(scenario.description)

    st.markdown("---")

    st.markdown("""
**How this app works:** Choose a hypothesis test from the sidebar and set the
population parameters (true effect size, variability, sample size, significance
level). The app computes statistical power — the probability of correctly
detecting the effect — either analytically via non-central distributions or
by Monte Carlo simulation. The plots show the null and true sampling
distributions, with the rejection region and power shaded.
""")

    st.markdown("---")

    st.markdown("""
**Statistical power** is the probability of correctly detecting a real effect
-- that is, rejecting the null hypothesis when the alternative is true.
It is the red shaded area in the plots below: the portion of the *true*
distribution that falls inside the rejection region.
""")

    sim_result = None
    if method == "Analytic":
        power = scenario.analytic_power(params)
    else:
        sim_result = scenario.simulate_power(params, n_simulations, seed)
        power = sim_result["power"]

    power_label = f"Power = {power:.4f}"
    if method == "Simulation":
        power_label += f"  ({n_simulations:,} simulations)"

    if power >= 0.80:
        st.success(power_label)
    elif power >= 0.50:
        st.warning(power_label)
    else:
        st.error(power_label)

    st.caption(
        f"Method: **{method}**. "
        "Convention: 80% power is a common target. "
        "Use the sliders to explore how effect size, variability, "
        "sample size, and significance level affect power."
    )

    st.markdown("---")
    render_distributions(scenario, params, method, sim_result)

    st.markdown("---")
    render_power_curve(scenario, params, method, n_simulations, seed)

    render_education()

    st.markdown("---")
    st.markdown(
        "**\\*** The test statistic follows a *non-central* distribution under H₁. "
        "The non-centrality parameter (NCP) measures how far the true distribution "
        "is shifted from the null -- larger NCP means more power."
    )
    st.latex(scenario.formula_latex())


if __name__ == "__main__":
    st.set_page_config(
        page_title="Statistical Power Explorer",
        page_icon="📊",
        layout="wide",
    )
    main()
