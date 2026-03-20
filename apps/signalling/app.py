"""Spence Signalling Model of Education — Streamlit app.

Explore the Spence (1973) job-market signalling model where education
serves not to increase productivity but to credibly signal pre-existing
ability. Compare separating, pooling, and first-best outcomes.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import streamlit as st

from shared.base import SliderSpec, EquilibriumConcept


class SpenceSignalling(EquilibriumConcept):
    """Spence job-market signalling with two worker types.

    Types:      H (high ability, productivity θ_H)
                L (low ability, productivity θ_L)
    Proportion: λ = Pr(H)
    Cost:       c_H · e for H types, c_L · e for L types
                Single-crossing: c_H < c_L

    Separating equilibrium (H gets e*, L gets 0):
        IC for H:  θ_H − c_H · e* ≥ θ_L          →  e* ≤ (θ_H − θ_L) / c_H
        IC for L:  θ_L ≥ θ_H − c_L · e*           →  e* ≥ (θ_H − θ_L) / c_L
        Feasible range: (θ_H − θ_L)/c_L  ≤  e*  ≤  (θ_H − θ_L)/c_H

    Pooling equilibrium:
        Both choose e = 0, wage = λ·θ_H + (1−λ)·θ_L
    """

    @property
    def name(self) -> str:
        return "Spence Signalling Model of Education"

    @property
    def description(self) -> str:
        return (
            "In Spence's (1973) model, education **does not raise productivity** "
            "— it serves purely as a **signal** of pre-existing ability. "
            "High-ability workers can afford to acquire costly education to "
            "distinguish themselves from low-ability workers, because the cost "
            "of education is lower for them. In a separating equilibrium, "
            "firms correctly infer ability from education, but education itself "
            "is socially wasteful."
        )

    def params(self) -> list[SliderSpec]:
        return [
            SliderSpec("theta_H", "High-type productivity (θ_H)", 5.0, 30.0, 20.0, 1.0,
                       "Output per period for high-ability workers"),
            SliderSpec("theta_L", "Low-type productivity (θ_L)", 1.0, 20.0, 10.0, 1.0,
                       "Output per period for low-ability workers"),
            SliderSpec("c_H", "High-type education cost (c_H)", 0.5, 10.0, 2.0, 0.5,
                       "Cost per unit of education for high-ability workers"),
            SliderSpec("c_L", "Low-type education cost (c_L)", 1.0, 15.0, 5.0, 0.5,
                       "Cost per unit of education for low-ability workers"),
            SliderSpec("lam", "Proportion of H types (λ)", 0.05, 0.95, 0.4, 0.05,
                       "Fraction of the workforce that is high-ability"),
            SliderSpec("e_star_frac", "Signal threshold (position in feasible range)",
                       0.0, 1.0, 0.5, 0.05,
                       "Where e* sits in the IC-compatible range (0 = min, 1 = max)"),
        ]

    def render(self, params: dict[str, Any], depth: str = "undergraduate") -> None:
        pass

    def compute_curves(self, params: dict[str, Any]) -> dict[str, Any]:
        return {}

    def compute_equilibrium(self, params: dict[str, Any]) -> dict[str, Any]:
        theta_H = params["theta_H"]
        theta_L = params["theta_L"]
        c_H = params["c_H"]
        c_L = params["c_L"]
        lam = params["lam"]
        frac = params["e_star_frac"]

        gap = theta_H - theta_L

        # Feasible range for e*
        e_min = gap / c_L
        e_max = gap / c_H

        # Check single-crossing (c_H < c_L) and θ_H > θ_L
        single_crossing = c_H < c_L
        valid = gap > 0 and single_crossing

        if valid:
            e_star = e_min + frac * (e_max - e_min)
        else:
            e_star = 0.0

        # --- Separating equilibrium ---
        # H type: chooses e*, gets θ_H, pays c_H · e*
        sep_payoff_H = theta_H - c_H * e_star if valid else None
        # L type: chooses 0, gets θ_L, pays nothing
        sep_payoff_L = theta_L

        # --- Pooling equilibrium ---
        pool_wage = lam * theta_H + (1 - lam) * theta_L
        pool_payoff_H = pool_wage
        pool_payoff_L = pool_wage

        # --- First-best (perfect information) ---
        fb_payoff_H = theta_H
        fb_payoff_L = theta_L

        # --- Social surplus ---
        # Separating: no education waste for L; H wastes c_H · e*
        # Total output is same (θ's unchanged), but resources spent on education
        sep_surplus = lam * theta_H + (1 - lam) * theta_L - lam * c_H * e_star if valid else None
        pool_surplus = lam * theta_H + (1 - lam) * theta_L  # no education cost
        fb_surplus = lam * theta_H + (1 - lam) * theta_L    # same as pooling in this model

        # Social cost of signalling
        signal_cost = lam * c_H * e_star if valid else None

        # IC slack
        ic_H_slack = (theta_H - c_H * e_star) - theta_L if valid else None
        ic_L_slack = (theta_H - c_L * e_star) - theta_L if valid else None
        # ic_L_slack should be ≤ 0 for L to prefer not signalling
        # (θ_H - c_L·e* - θ_L ≤ 0)

        # Pareto dominance: does pooling make BOTH types at least as well off?
        # L always prefers pooling (pool_wage > θ_L when λ > 0).
        # H prefers pooling when pool_wage > sep_payoff_H.
        if valid:
            h_prefers_pooling = pool_wage > sep_payoff_H + 1e-10
            pareto_dominated = h_prefers_pooling  # L already prefers pooling
        else:
            h_prefers_pooling = None
            pareto_dominated = None

        return {
            "valid": valid,
            "single_crossing": single_crossing,
            "gap": gap,
            "e_min": e_min,
            "e_max": e_max,
            "e_star": e_star,
            # Separating
            "sep_payoff_H": sep_payoff_H,
            "sep_payoff_L": sep_payoff_L,
            # Pooling
            "pool_wage": pool_wage,
            "pool_payoff_H": pool_payoff_H,
            "pool_payoff_L": pool_payoff_L,
            # First-best
            "fb_payoff_H": fb_payoff_H,
            "fb_payoff_L": fb_payoff_L,
            # Surplus
            "sep_surplus": sep_surplus,
            "pool_surplus": pool_surplus,
            "fb_surplus": fb_surplus,
            "signal_cost": signal_cost,
            # IC
            "ic_H_slack": ic_H_slack,
            "ic_L_slack": ic_L_slack,
            "h_prefers_pooling": h_prefers_pooling,
            "pareto_dominated": pareto_dominated,
        }

    def educational_sections(self, depth: str = "undergraduate") -> list:
        sections = [
            ("Why education can be pure waste",
             "In the human capital model (Becker, 1964), education raises "
             "your productivity — it's an investment. In the signalling "
             "model (Spence, 1973), education **doesn't change what you can "
             "produce**. It only serves to separate the high-ability workers "
             "from the low-ability ones.\n\n"
             "If firms could observe ability directly, nobody would need "
             "education at all — wages would just equal productivity. "
             "Education exists only because of the information asymmetry."),
            ("The single-crossing condition",
             "The model requires that education is **cheaper for high-ability "
             "workers** (c_H < c_L). This is the *single-crossing* or "
             "*Spence-Mirrlees* condition.\n\n"
             "Why might this hold?\n"
             "- Smarter students find coursework easier\n"
             "- Motivated workers handle the grind of school better\n"
             "- Higher-ability people finish degrees faster\n\n"
             "Without single-crossing, low-ability workers could mimic "
             "high-ability ones at the same cost, and signalling would "
             "break down."),
            ("Multiple equilibria",
             "Any e* in the range [(θ_H − θ_L)/c_L, (θ_H − θ_L)/c_H] "
             "supports a separating equilibrium. This is a **continuum of "
             "equilibria** — the model alone doesn't pin down which one "
             "prevails.\n\n"
             "- The **minimum** e* (the Riley outcome) is most efficient — "
             "least signalling waste\n"
             "- The **maximum** e* is the most costly — H types are "
             "indifferent between signalling and not\n\n"
             "Refinements like the *Intuitive Criterion* (Cho & Kreps, 1987) "
             "select the Riley outcome as the unique 'reasonable' equilibrium."),
            ("Signalling vs human capital",
             "Reality likely involves **both** mechanisms:\n"
             "- Education partly raises productivity (human capital)\n"
             "- Education partly signals pre-existing ability (signalling)\n\n"
             "The policy implications differ sharply:\n"
             "- If human capital: subsidise education → more productivity\n"
             "- If signalling: subsidise education → more waste\n\n"
             "Empirically distinguishing the two is one of the hardest "
             "problems in labour economics."),
        ]
        if depth == "graduate":
            sections.append((
                "Formal equilibrium conditions",
                "A **separating Perfect Bayesian Equilibrium** consists of:\n\n"
                "1. **Strategies**: H chooses e = e*, L chooses e = 0\n"
                "2. **Beliefs**: firms believe Pr(H | e ≥ e*) = 1, "
                "Pr(H | e < e*) = 0\n"
                "3. **Wages**: w(e) = θ_H if e ≥ e*, w(e) = θ_L otherwise\n\n"
                "**Incentive compatibility:**\n\n"
                r"$$\theta_H - c_H e^* \geq \theta_L \quad \Rightarrow \quad "
                r"e^* \leq \frac{\theta_H - \theta_L}{c_H}$$"
                "\n"
                r"$$\theta_L \geq \theta_H - c_L e^* \quad \Rightarrow \quad "
                r"e^* \geq \frac{\theta_H - \theta_L}{c_L}$$"
                "\n\nA **pooling equilibrium** has both types choose "
                "the same e (typically 0), with wage equal to the prior "
                "expected productivity λθ_H + (1−λ)θ_L. Pooling equilibria "
                "fail the Intuitive Criterion when single-crossing holds."
            ))
        return sections


# ---------------------------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------------------------

def main():
    concept = SpenceSignalling()

    # --- Sidebar ---
    st.sidebar.header("Spence Signalling")
    st.sidebar.markdown("---")

    st.sidebar.subheader("Productivity")
    theta_H = st.sidebar.slider("High-type productivity (θ_H)",
                                5.0, 30.0, 20.0, 1.0)
    theta_L = st.sidebar.slider("Low-type productivity (θ_L)",
                                1.0, 20.0, 10.0, 1.0)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Education costs")
    c_H = st.sidebar.slider("High-type cost (c_H)", 0.5, 10.0, 2.0, 0.5,
                            help="Must be less than c_L for single-crossing")
    c_L = st.sidebar.slider("Low-type cost (c_L)", 1.0, 15.0, 5.0, 0.5,
                            help="Must be greater than c_H for single-crossing")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Population")
    lam = st.sidebar.slider("Proportion of H types (λ)", 0.05, 0.95, 0.4, 0.05)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Equilibrium selection")
    e_star_frac = st.sidebar.slider(
        "Signal threshold (e*) position", 0.0, 1.0, 0.5, 0.05,
        help="0 = minimum viable signal (Riley), 1 = maximum")

    st.sidebar.markdown("---")
    depth = st.sidebar.radio("Depth", ["undergraduate", "graduate"],
                             format_func=lambda x: x.title())

    p = {"theta_H": theta_H, "theta_L": theta_L, "c_H": c_H, "c_L": c_L,
         "lam": lam, "e_star_frac": e_star_frac}
    eq = concept.compute_equilibrium(p)

    # --- Main area ---
    st.title("Spence Signalling Model of Education")
    st.markdown(concept.description)

    st.markdown("---")

    st.markdown("""
**How this app works:** Two types of workers (high and low ability) differ
in productivity and in the cost of acquiring education. Firms cannot observe
ability directly but can observe education. The sidebar sets productivities,
education costs, the fraction of high types, and where the signalling
threshold e* sits within the incentive-compatible range. The app compares
the separating equilibrium (high types signal, low types don't) with
the pooling equilibrium (no signalling, firms pay expected productivity)
and the first-best (perfect information).
""")

    st.markdown("---")

    # Validity checks
    if theta_H <= theta_L:
        st.error("θ_H must exceed θ_L for there to be an information problem.")
        return
    if c_H >= c_L:
        st.error("Single-crossing requires c_H < c_L — high-ability workers "
                 "must find education cheaper.")
        return

    # --- Headline ---
    gap = eq["gap"]
    st.markdown(f"**Separating equilibrium** — feasible signal range: "
                f"e* ∈ [{eq['e_min']:.2f}, {eq['e_max']:.2f}], "
                f"current e* = {eq['e_star']:.2f}")

    # --- Plots ---

    # 1. Spence signalling diagram (full width)
    st.markdown("**Signalling Diagram**")
    fig1, ax1 = plt.subplots(figsize=(9, 5.5))

    e_plot_max = eq["e_max"] * 1.6
    e_range = np.linspace(0, e_plot_max, 500)
    e_star = eq["e_star"]

    # Wage schedule (step function)
    wage_schedule = np.where(e_range >= e_star, theta_H, theta_L)
    ax1.step(e_range, wage_schedule, where="post", color="#2563eb",
             linewidth=2.5, label="Wage schedule w(e)", zorder=3)

    # Shade the feasible e* range
    ax1.axvspan(eq["e_min"], eq["e_max"], alpha=0.08, color="#7c3aed",
                label=f"Feasible e* range")

    # Cost lines from the origin payoff (starting at θ_L with slope c)
    # If type chooses e = 0, gets θ_L. If chooses e ≥ e*, gets θ_H.
    # Net payoff of NOT signalling = θ_L
    # Net payoff of signalling at e = θ_H - c·e
    # Indifference: θ_H - c·e shown as downward-sloping line from θ_H

    # H type's payoff if signalling: θ_H - c_H·e (line starting at θ_H)
    payoff_H = theta_H - c_H * e_range
    ax1.plot(e_range, payoff_H, color="#dc2626", linewidth=2,
             label=f"H payoff: θ_H − c_H·e  (c_H={c_H})")

    # L type's payoff if signalling: θ_H - c_L·e (steeper line)
    payoff_L = theta_H - c_L * e_range
    ax1.plot(e_range, payoff_L, color="#f59e0b", linewidth=2,
             label=f"L payoff: θ_H − c_L·e  (c_L={c_L})")

    # Horizontal lines for no-signal payoffs
    ax1.axhline(theta_L, color="#64748b", linewidth=1, linestyle=":",
                alpha=0.6)
    ax1.axhline(theta_H, color="#64748b", linewidth=1, linestyle=":",
                alpha=0.6)

    # Labels for θ_H and θ_L
    ax1.text(-0.15 * e_plot_max, theta_H, f"θ_H = {theta_H:.0f}",
             va="center", ha="right", fontsize=10, color="#64748b",
             fontweight="bold")
    ax1.text(-0.15 * e_plot_max, theta_L, f"θ_L = {theta_L:.0f}",
             va="center", ha="right", fontsize=10, color="#64748b",
             fontweight="bold")

    # Mark e* with vertical line
    ax1.axvline(e_star, color="#7c3aed", linewidth=2, linestyle="--",
                alpha=0.7)
    ax1.text(e_star, theta_H * 1.05, f"e* = {e_star:.2f}",
             ha="center", va="bottom", fontsize=10, color="#7c3aed",
             fontweight="bold")

    # Mark the equilibrium points
    # H type at e*: gets θ_H, net = θ_H - c_H·e*
    ax1.plot(e_star, eq["sep_payoff_H"], "o", color="#dc2626",
             markersize=11, zorder=6)
    ax1.text(e_star + 0.15 * e_plot_max * 0.1, eq["sep_payoff_H"],
             f"  H: {eq['sep_payoff_H']:.1f}",
             va="center", fontsize=9, color="#dc2626", fontweight="bold")

    # L type at e=0: gets θ_L
    ax1.plot(0, theta_L, "s", color="#f59e0b", markersize=11, zorder=6)
    ax1.text(0.3, theta_L - 0.5, f"  L: {theta_L:.1f}",
             va="top", fontsize=9, color="#f59e0b", fontweight="bold")

    # Mark e_min and e_max on axis
    ax1.plot(eq["e_min"], theta_L, "|", color="#7c3aed", markersize=15,
             markeredgewidth=2, zorder=5)
    ax1.plot(eq["e_max"], theta_L, "|", color="#7c3aed", markersize=15,
             markeredgewidth=2, zorder=5)
    ax1.text(eq["e_min"], theta_L * 0.85, f"e_min\n{eq['e_min']:.2f}",
             ha="center", va="top", fontsize=8, color="#7c3aed")
    ax1.text(eq["e_max"], theta_L * 0.85, f"e_max\n{eq['e_max']:.2f}",
             ha="center", va="top", fontsize=8, color="#7c3aed")

    ax1.set_xlabel("Education level (e)", fontsize=11)
    ax1.set_ylabel("Wage / Net payoff", fontsize=11)
    ax1.set_xlim(-0.15 * e_plot_max, e_plot_max)
    ax1.set_ylim(0, theta_H * 1.15)
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(True, alpha=0.2)
    ax1.set_title(
        "H type signals (chooses e*), L type doesn't (chooses 0)",
        fontsize=11, fontweight="bold")
    fig1.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)

    # 2–3. Payoff comparison + welfare (side by side)
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**Payoff Comparison by Type**")
        fig2, ax2 = plt.subplots(figsize=(6, 4.5))

        x = np.arange(3)
        width = 0.3

        h_payoffs = [eq["sep_payoff_H"], eq["pool_payoff_H"], eq["fb_payoff_H"]]
        l_payoffs = [eq["sep_payoff_L"], eq["pool_payoff_L"], eq["fb_payoff_L"]]

        bars_h = ax2.bar(x - width / 2, h_payoffs, width, color="#dc2626",
                         edgecolor="white", label="H type")
        bars_l = ax2.bar(x + width / 2, l_payoffs, width, color="#f59e0b",
                         edgecolor="white", label="L type")

        for bar, val in zip(list(bars_h) + list(bars_l),
                            h_payoffs + l_payoffs):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{val:.1f}", ha="center", va="bottom",
                     fontsize=9, fontweight="bold")

        ax2.set_xticks(x)
        ax2.set_xticklabels(["Separating", "Pooling", "First-best"],
                            fontsize=9)
        ax2.set_ylabel("Net payoff", fontsize=10)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.2, axis="y")
        ax2.set_title("Who gains from signalling?", fontsize=12,
                      fontweight="bold")
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    with col_right:
        st.markdown("**Social Surplus**")
        fig3, ax3 = plt.subplots(figsize=(6, 4.5))

        regimes = ["Separating", "Pooling\n(= First-best)"]
        surpluses = [eq["sep_surplus"], eq["pool_surplus"]]
        colors = ["#7c3aed", "#2563eb"]

        bars = ax3.bar(regimes, surpluses, color=colors, edgecolor="white",
                       width=0.4)
        for bar, val in zip(bars, surpluses):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{val:.1f}", ha="center", va="bottom",
                     fontsize=11, fontweight="bold")

        # Show the waste
        if eq["signal_cost"] > 0:
            ax3.bar(["Separating"], [eq["signal_cost"]],
                    bottom=[eq["sep_surplus"]], color="#dc2626", alpha=0.4,
                    edgecolor="white", width=0.4,
                    label=f"Signalling waste: {eq['signal_cost']:.1f}")
            ax3.legend(fontsize=9, loc="lower right")

        ax3.set_ylabel("Total surplus (per worker)", fontsize=10)
        ax3.grid(True, alpha=0.2, axis="y")
        ax3.set_ylim(0, eq["pool_surplus"] * 1.2)
        ax3.set_title("Signalling is socially wasteful",
                      fontsize=12, fontweight="bold")
        fig3.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

    st.caption(
        "**Top:** The signalling diagram shows the wage schedule (step function "
        "at e*) and the payoff-if-signalling lines for each type. H type's "
        "line is flatter (lower cost) so signalling is worthwhile for H but "
        "not for L. The purple region marks all incentive-compatible values of "
        "e*. "
        "**Bottom left:** Payoffs for each type under three regimes. "
        "L types always prefer pooling (the average wage exceeds θ_L). "
        "Whether H types prefer separating depends on λ and e*. "
        "**Bottom right:** Total surplus is highest without signalling — "
        "education is pure social waste in this model."
    )

    # --- Pareto dominance callout ---
    if eq["pareto_dominated"]:
        st.info(
            f"**Pooling Pareto-dominates separating here.** "
            f"H types earn {eq['sep_payoff_H']:.1f} in the separating "
            f"equilibrium but would earn {eq['pool_wage']:.1f} under "
            f"pooling — and L types are also better off under pooling "
            f"({eq['pool_wage']:.1f} vs {eq['sep_payoff_L']:.1f}). "
            f"Both types would prefer to abolish signalling, yet the "
            f"separating equilibrium is sustained because no *individual* "
            f"can deviate: if an H type drops education to zero, firms "
            f"infer they are L type and pay only θ_L = {theta_L:.0f}. "
            f"The equilibrium is held in place by off-equilibrium beliefs, "
            f"not by anyone's preference for it.\n\n"
            f"Try lowering λ or moving e* toward the minimum — "
            f"H types are more likely to prefer separating when the pooling "
            f"wage is low or signalling costs are small."
        )
    elif eq["h_prefers_pooling"] is not None and not eq["h_prefers_pooling"]:
        st.success(
            f"**H types prefer separating** "
            f"({eq['sep_payoff_H']:.1f} > {eq['pool_wage']:.1f} pooling wage). "
            f"L types still prefer pooling — separating is not a Pareto "
            f"improvement over pooling, but H types actively benefit from "
            f"being able to distinguish themselves."
        )

    st.markdown("---")

    # --- Metrics ---
    st.markdown("**Equilibrium Values**")
    c1, c2, c3 = st.columns(3)

    c1.markdown("*Separating*")
    c1.text(f"  e* = {e_star:.2f}\n"
            f"  H wage = {theta_H:.1f},  net = {eq['sep_payoff_H']:.1f}\n"
            f"  L wage = {theta_L:.1f},  net = {theta_L:.1f}\n"
            f"  Signal cost = {eq['signal_cost']:.1f}")

    c2.markdown("*Pooling*")
    c2.text(f"  e = 0 (both types)\n"
            f"  Wage = {eq['pool_wage']:.1f}  (= λθ_H + (1-λ)θ_L)\n"
            f"  H net = {eq['pool_payoff_H']:.1f}\n"
            f"  L net = {eq['pool_payoff_L']:.1f}")

    c3.markdown("*First-best*")
    c3.text(f"  No education needed\n"
            f"  H wage = {theta_H:.1f}\n"
            f"  L wage = {theta_L:.1f}\n"
            f"  Perfect information")

    st.markdown("---")
    st.markdown("**Incentive Compatibility**")
    c1, c2 = st.columns(2)
    c1.metric("H prefers signalling?",
              f"{'Yes' if eq['ic_H_slack'] >= -1e-10 else 'No'}"
              f"  (slack = {eq['ic_H_slack']:+.2f})",
              help="θ_H − c_H·e* ≥ θ_L")
    c2.metric("L prefers NOT signalling?",
              f"{'Yes' if eq['ic_L_slack'] <= 1e-10 else 'No'}"
              f"  (slack = {eq['ic_L_slack']:+.2f})",
              help="θ_L ≥ θ_H − c_L·e*  ⟺  θ_H − c_L·e* − θ_L ≤ 0")

    # Educational sections
    sections = concept.educational_sections(depth)
    if sections:
        st.markdown("---")
        st.subheader("Learn More")
        for title, body in sections:
            with st.expander(title):
                st.markdown(body)


if __name__ == "__main__":
    st.set_page_config(
        page_title="Spence Signalling Model",
        page_icon="📊",
        layout="wide",
    )
    main()
