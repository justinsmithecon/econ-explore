"""Income and Substitution Effects in Labour Supply — Streamlit app.

Decompose the effect of a wage change on labour supply into
a substitution effect (leisure becomes more/less expensive)
and an income effect (the worker is richer/poorer).
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from shared.base import SliderSpec, EquilibriumConcept


class LabourSupply(EquilibriumConcept):
    """Income and substitution effects with Cobb-Douglas preferences.

    Utility:    U(C, ℓ) = C^β · ℓ^(1-β)
    Budget:     C = w·(T - ℓ) + V   ⟹   C + w·ℓ = wT + V  (full income)
    Time:       T = total hours,  ℓ = leisure,  h = T - ℓ = work hours

    Marshallian demands:
        ℓ* = (1-β)(wT + V) / w
        C* = β(wT + V)

    Hicksian (compensated) leisure at wage w for utility Ū:
        ℓ^h = Ū · ((1-β) / (βw))^β
    """

    @property
    def name(self) -> str:
        return "Income & Substitution Effects in Labour Supply"

    @property
    def description(self) -> str:
        return (
            "When wages rise, two opposing forces shape how much a worker "
            "chooses to work. The **substitution effect**: leisure is now "
            "more expensive (higher opportunity cost), so the worker "
            "substitutes toward more work and less leisure. The **income "
            "effect**: the worker is effectively richer, and since leisure "
            "is a normal good, they want more of it — meaning less work. "
            "The net effect is ambiguous, which is why the labour supply "
            "curve can bend backward at high wages."
        )

    def params(self) -> list[SliderSpec]:
        return [
            SliderSpec("beta", "Consumption weight (β)", 0.1, 0.9, 0.5, 0.05,
                       "Exponent on consumption in U = C^β · ℓ^(1-β)"),
            SliderSpec("T", "Time endowment (T)", 10.0, 24.0, 16.0, 1.0,
                       "Total hours available to split between work and leisure"),
            SliderSpec("w_0", "Initial wage (w₀)", 1.0, 30.0, 10.0, 0.5,
                       "Starting hourly wage"),
            SliderSpec("w_1", "New wage (w₁)", 1.0, 30.0, 15.0, 0.5,
                       "Wage after the change"),
            SliderSpec("V", "Non-labour income (V)", 0.0, 100.0, 20.0, 5.0,
                       "Income from sources other than work"),
        ]

    def render(self, params: dict[str, Any], depth: str = "undergraduate") -> None:
        pass

    def _marshallian(self, w: float, beta: float, T: float,
                     V: float) -> tuple[float, float]:
        """Marshallian (uncompensated) optimal leisure and consumption."""
        full_income = w * T + V
        leisure = (1 - beta) * full_income / w
        consumption = beta * full_income
        return leisure, consumption

    def _utility(self, C: float, leisure: float,
                 beta: float) -> float:
        """Cobb-Douglas utility."""
        if C <= 0 or leisure <= 0:
            return 0.0
        return C ** beta * leisure ** (1 - beta)

    def _hicksian_leisure(self, w: float, U_bar: float,
                          beta: float) -> float:
        """Compensated (Hicksian) leisure demand at wage w for utility Ū."""
        return U_bar * ((1 - beta) / (beta * w)) ** beta

    def _hicksian_consumption(self, w: float, leisure_h: float,
                              beta: float) -> float:
        """Consumption at the compensated point."""
        return beta * w * leisure_h / (1 - beta)

    def compute_curves(self, params: dict[str, Any]) -> dict[str, Any]:
        return {}

    def compute_equilibrium(self, params: dict[str, Any]) -> dict[str, Any]:
        beta = params["beta"]
        T = params["T"]
        w_0, w_1 = params["w_0"], params["w_1"]
        V = params["V"]

        # A: Initial optimum at w₀
        ell_0, C_0 = self._marshallian(w_0, beta, T, V)
        h_0 = T - ell_0
        U_0 = self._utility(C_0, ell_0, beta)

        # B: Compensated (Hicksian) at w₁, holding utility at U₀
        ell_comp = self._hicksian_leisure(w_1, U_0, beta)
        C_comp = self._hicksian_consumption(w_1, ell_comp, beta)
        h_comp = T - ell_comp

        # C: Final optimum at w₁
        ell_1, C_1 = self._marshallian(w_1, beta, T, V)
        h_1 = T - ell_1
        U_1 = self._utility(C_1, ell_1, beta)

        # Decomposition (in terms of hours worked)
        substitution_effect = h_comp - h_0   # always positive for wage increase
        income_effect = h_1 - h_comp          # negative for wage increase (leisure normal)
        total_effect = h_1 - h_0

        # Compensating variation: extra income needed at w₁ to reach U₀
        # Full income at compensated point
        full_income_comp = w_1 * ell_comp + C_comp
        compensating_variation = full_income_comp - (w_1 * T + V)

        return {
            # Initial (A)
            "ell_0": ell_0, "C_0": C_0, "h_0": h_0, "U_0": U_0,
            # Compensated (B)
            "ell_comp": ell_comp, "C_comp": C_comp, "h_comp": h_comp,
            # Final (C)
            "ell_1": ell_1, "C_1": C_1, "h_1": h_1, "U_1": U_1,
            # Decomposition
            "substitution_effect": substitution_effect,
            "income_effect": income_effect,
            "total_effect": total_effect,
            "compensating_variation": compensating_variation,
        }

    def labour_supply_curve(self, beta: float, T: float, V: float,
                            w_min: float = 0.5,
                            w_max: float = 40.0) -> tuple[np.ndarray, np.ndarray]:
        """Compute the labour supply curve h*(w) over a wage range."""
        wages = np.linspace(w_min, w_max, 300)
        hours = np.array([T - (1 - beta) * (w * T + V) / w for w in wages])
        return wages, hours

    def educational_sections(self) -> list:
        sections = [
            ("Substitution effect",
             "When the wage rises, each hour of leisure now costs more in "
             "forgone earnings. The worker is incentivised to **substitute** "
             "away from leisure and toward consumption (i.e. work more). "
             "On the diagram, this is the movement along the original "
             "indifference curve from A to B — the budget line has rotated "
             "but we compensate the worker to stay on the same utility "
             "level.\n\n"
             "The substitution effect always increases hours worked when "
             "the wage rises (and vice versa)."),
            ("Income effect",
             "A higher wage also makes the worker **richer** — they can "
             "afford more of everything, including leisure. Since leisure "
             "is a normal good, the income effect pushes toward more "
             "leisure and fewer hours worked.\n\n"
             "On the diagram, this is the parallel shift from B to C — "
             "same slope (same relative prices) but a higher budget. The "
             "income effect works **against** the substitution effect."),
            ("The backward-bending supply curve",
             "At low wages, the substitution effect dominates: workers "
             "are eager to trade cheap leisure for consumption. The supply "
             "curve slopes upward.\n\n"
             "At high wages, the income effect can dominate: workers are "
             "already rich enough and value leisure more at the margin. "
             "The supply curve bends backward.\n\n"
             "With Cobb-Douglas preferences the bend-back occurs when "
             "non-labour income V > 0, because the income effect from "
             "being wealthier matters more. With V = 0 and Cobb-Douglas, "
             "hours are constant at h = βT — the two effects exactly "
             "cancel."),
            ("Contrast with labour demand",
             "For labour **demand**, substitution and scale effects both "
             "reduce employment when wages rise — making the demand curve "
             "unambiguously downward-sloping.\n\n"
             "For labour **supply**, the substitution and income effects "
             "push in opposite directions. This asymmetry is why we can "
             "have a backward-bending supply curve but never an upward-"
             "sloping demand curve."),
        ]
        sections.append((
                "Slutsky equation for labour supply (Advanced)",
                "The Slutsky equation decomposes the wage effect on hours:\n\n"
                r"$$\frac{\partial h}{\partial w} = "
                r"\underbrace{\frac{\partial h^c}{\partial w}}_{\text{substitution}} "
                r"+ \underbrace{h \cdot \frac{\partial h}{\partial V}}_{\text{income}}$$"
                "\n\nWith Cobb-Douglas U = C^β ℓ^(1-β), the Marshallian demands "
                "are:\n\n"
                r"$$\ell^* = \frac{(1-\beta)(wT + V)}{w}, \quad "
                r"C^* = \beta(wT + V)$$"
                "\n\nThe Hicksian (compensated) leisure demand, holding utility "
                "at Ū, is:\n\n"
                r"$$\ell^h(w, \bar{U}) = \bar{U} \left(\frac{1-\beta}{\beta w}"
                r"\right)^\beta$$"
                "\n\nSince ∂ℓ^h/∂w < 0, the compensated response always reduces "
                "leisure (increases work). The income effect ∂h/∂V = (1-β)/w > 0 "
                "for leisure, so higher income → more leisure → less work, "
                "opposing the substitution effect."
            ))
        return sections


# ---------------------------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------------------------

def main():
    concept = LabourSupply()

    # --- Sidebar ---
    st.sidebar.header("Labour Supply")
    st.sidebar.markdown("---")

    st.sidebar.subheader("Preferences")
    beta = st.sidebar.slider("Consumption weight (β)", 0.1, 0.9, 0.5, 0.05,
                             help="U = C^β · ℓ^(1-β)")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Time & income")
    T = st.sidebar.slider("Time endowment (T)", 10.0, 24.0, 16.0, 1.0,
                          help="Total hours to split between work and leisure")
    V = st.sidebar.slider("Non-labour income (V)", 0.0, 100.0, 20.0, 5.0,
                          help="Income from sources other than work")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Wage change")
    w_0 = st.sidebar.slider("Initial wage (w₀)", 1.0, 30.0, 10.0, 0.5)
    w_1 = st.sidebar.slider("New wage (w₁)", 1.0, 30.0, 15.0, 0.5)

    p = {"beta": beta, "T": T, "w_0": w_0, "w_1": w_1, "V": V}
    eq = concept.compute_equilibrium(p)

    # --- Main area ---
    st.title("Income & Substitution Effects in Labour Supply")
    st.markdown(concept.description)

    st.markdown("---")

    wage_dir = "increase" if w_1 > w_0 else "decrease" if w_1 < w_0 else "change"
    st.markdown(f"""
**How this app works:** A worker splits their time between work and leisure,
with Cobb-Douglas preferences U = C^β · ℓ^(1-β). The sidebar sets the
preference weight on consumption, the time endowment, non-labour income,
and an initial and new wage. The app decomposes the effect of the wage
{wage_dir} on hours worked into a substitution effect (re-optimising along
the original indifference curve at the new wage) and an income effect (the
parallel shift to the final budget line). The labour supply curve on the
right shows how hours vary across all wages.
""")

    st.markdown("---")

    # --- Plots ---

    # 1. Indifference curve / budget constraint diagram (full width)
    st.markdown("**Consumption–Leisure Diagram (Slutsky Decomposition)**")
    fig1, ax1 = plt.subplots(figsize=(9, 5.5))

    # Axis ranges
    ell_max = T * 1.15
    C_max = max(eq["C_0"], eq["C_comp"], eq["C_1"]) * 1.6

    ell_range = np.linspace(0.01, T, 500)

    # Budget constraints
    # Original: C = w₀(T - ℓ) + V = w₀T + V - w₀ℓ
    C_budget_0 = w_0 * (T - ell_range) + V
    ax1.plot(ell_range[C_budget_0 >= 0], C_budget_0[C_budget_0 >= 0],
             color="#dc2626", linewidth=1.8, alpha=0.7,
             label=f"Budget w₀={w_0:.1f}")

    # Final: C = w₁(T - ℓ) + V
    C_budget_1 = w_1 * (T - ell_range) + V
    ax1.plot(ell_range[C_budget_1 >= 0], C_budget_1[C_budget_1 >= 0],
             color="#16a34a", linewidth=1.8, alpha=0.7,
             label=f"Budget w₁={w_1:.1f}")

    # Compensated budget: passes through B with slope -w₁
    # C = C_comp + w₁(ell_comp - ℓ)
    C_budget_comp = eq["C_comp"] + w_1 * (eq["ell_comp"] - ell_range)
    ax1.plot(ell_range[C_budget_comp >= 0], C_budget_comp[C_budget_comp >= 0],
             color="#f59e0b", linewidth=1.5, linestyle="--", alpha=0.7,
             label="Compensated budget")

    # Indifference curves
    # U = C^β · ℓ^(1-β) → C = (U / ℓ^(1-β))^(1/β)
    ell_ic = np.linspace(0.01, ell_max, 500)

    C_ic_0 = (eq["U_0"] / ell_ic ** (1 - beta)) ** (1 / beta)
    valid = C_ic_0 < C_max * 1.2
    ax1.plot(ell_ic[valid], C_ic_0[valid],
             color="#2563eb", linewidth=2, label=f"U₀ = {eq['U_0']:.1f}")

    C_ic_1 = (eq["U_1"] / ell_ic ** (1 - beta)) ** (1 / beta)
    valid = C_ic_1 < C_max * 1.2
    ax1.plot(ell_ic[valid], C_ic_1[valid],
             color="#2563eb", linewidth=1.5, linestyle="--",
             label=f"U₁ = {eq['U_1']:.1f}")

    # Points A, B, C
    ax1.plot(eq["ell_0"], eq["C_0"], "o", color="#dc2626", markersize=11,
             zorder=6, label="A (initial)")
    ax1.plot(eq["ell_comp"], eq["C_comp"], "s", color="#f59e0b", markersize=11,
             zorder=6, label="B (substitution)")
    ax1.plot(eq["ell_1"], eq["C_1"], "^", color="#16a34a", markersize=11,
             zorder=6, label="C (final)")

    # Arrows A→B and B→C
    ax1.annotate("", xy=(eq["ell_comp"], eq["C_comp"]),
                 xytext=(eq["ell_0"], eq["C_0"]),
                 arrowprops=dict(arrowstyle="->", color="#7c3aed", lw=2.5))
    ax1.annotate("", xy=(eq["ell_1"], eq["C_1"]),
                 xytext=(eq["ell_comp"], eq["C_comp"]),
                 arrowprops=dict(arrowstyle="->", color="#16a34a", lw=2.5))

    ax1.set_xlabel("Leisure (ℓ)", fontsize=11)
    ax1.set_ylabel("Consumption (C)", fontsize=11)
    ax1.set_xlim(0, ell_max)
    ax1.set_ylim(0, C_max)
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(True, alpha=0.2)
    ax1.set_title(
        "A → B: Substitution (along indifference curve)    "
        "B → C: Income (parallel shift)",
        fontsize=11, fontweight="bold")
    fig1.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)

    # 2–3. Labour supply curve + decomposition bar (side by side)
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**Labour Supply Curve**")
        fig2, ax2 = plt.subplots(figsize=(6, 4.5))

        wages, hours = concept.labour_supply_curve(beta, T, V)
        # Clip to non-negative hours
        valid = hours >= 0
        ax2.plot(hours[valid], wages[valid], color="#2563eb", linewidth=2,
                 label="h*(w)")

        # Mark initial and final points
        ax2.plot(eq["h_0"], w_0, "o", color="#dc2626", markersize=10, zorder=6,
                 label=f"A: h={eq['h_0']:.1f}")
        ax2.plot(eq["h_1"], w_1, "^", color="#16a34a", markersize=10, zorder=6,
                 label=f"C: h={eq['h_1']:.1f}")

        # Arrow from initial to final
        if abs(eq["h_1"] - eq["h_0"]) > 0.05:
            ax2.annotate("", xy=(eq["h_1"], w_1), xytext=(eq["h_0"], w_0),
                         arrowprops=dict(arrowstyle="->", color="#7c3aed",
                                         lw=2, connectionstyle="arc3,rad=0.2"))

        ax2.set_xlabel("Hours worked (h)", fontsize=10)
        ax2.set_ylabel("Wage (w)", fontsize=10)
        ax2.set_xlim(left=0)
        ax2.set_ylim(bottom=0)
        ax2.legend(fontsize=8, loc="best")
        ax2.grid(True, alpha=0.2)
        backward = (w_1 > w_0 and eq["h_1"] < eq["h_0"]) or \
                   (w_1 < w_0 and eq["h_1"] > eq["h_0"])
        title = "Backward-bending here!" if backward else "Supply slopes upward here"
        ax2.set_title(title, fontsize=12, fontweight="bold")
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    with col_right:
        st.markdown("**Decomposition of Δh**")
        fig3, ax3 = plt.subplots(figsize=(6, 4.5))

        labels = ["Total", "Substitution", "Income"]
        values = [eq["total_effect"], eq["substitution_effect"],
                  eq["income_effect"]]
        colors = ["#64748b", "#7c3aed", "#16a34a"]

        bars = ax3.bar(labels, values, color=colors, edgecolor="white", width=0.5)
        for bar, val in zip(bars, values):
            y = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2, y,
                     f"{val:+.2f}", ha="center",
                     va="bottom" if y >= 0 else "top",
                     fontsize=11, fontweight="bold")

        ax3.axhline(0, color="black", linewidth=0.5)
        ax3.set_ylabel("Change in hours worked (Δh)", fontsize=10)
        ax3.grid(True, alpha=0.2, axis="y")
        ax3.set_title("Decomposition", fontsize=12, fontweight="bold")
        fig3.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

    st.caption(
        "**Top:** A → B is the substitution effect (along the U₀ indifference "
        "curve — same utility, new price ratio). B → C is the income effect "
        "(parallel shift to the final budget line). "
        "**Bottom left:** The labour supply curve shows hours worked at each "
        "wage — it can bend backward when the income effect dominates. "
        "**Bottom right:** The bar chart decomposes the total change in hours."
    )

    st.markdown("---")

    # --- Metrics ---
    st.markdown("**Equilibrium Values**")
    c1, c2, c3 = st.columns(3)
    c1.markdown("*Initial (A)*")
    c1.text(f"  Leisure  = {eq['ell_0']:.2f}\n"
            f"  Hours    = {eq['h_0']:.2f}\n"
            f"  Consump. = {eq['C_0']:.2f}\n"
            f"  Utility  = {eq['U_0']:.2f}")
    c2.markdown("*Compensated (B)*")
    c2.text(f"  Leisure  = {eq['ell_comp']:.2f}\n"
            f"  Hours    = {eq['h_comp']:.2f}\n"
            f"  Consump. = {eq['C_comp']:.2f}\n"
            f"  Utility  = {eq['U_0']:.2f}")
    c3.markdown("*Final (C)*")
    c3.text(f"  Leisure  = {eq['ell_1']:.2f}\n"
            f"  Hours    = {eq['h_1']:.2f}\n"
            f"  Consump. = {eq['C_1']:.2f}\n"
            f"  Utility  = {eq['U_1']:.2f}")

    st.markdown("---")
    st.markdown("**Decomposition**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Substitution effect", f"{eq['substitution_effect']:+.2f}")
    c2.metric("Income effect", f"{eq['income_effect']:+.2f}")
    c3.metric("Total effect", f"{eq['total_effect']:+.2f}")

    # Educational sections
    sections = concept.educational_sections()
    if sections:
        st.markdown("---")
        st.subheader("Learn More")
        for title, body in sections:
            with st.expander(title):
                st.markdown(body)


if __name__ == "__main__":
    st.set_page_config(
        page_title="Labour Supply — Income & Substitution",
        page_icon="📊",
        layout="wide",
    )
    main()
