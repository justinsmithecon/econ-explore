"""Substitution and Scale Effects in Labour Demand — Streamlit app.

Decompose the effect of a wage change on labour demand into
a substitution effect (changing the input mix along an isoquant)
and a scale effect (changing the level of output).
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from shared.base import SliderSpec, EquilibriumConcept


class LabourDemand(EquilibriumConcept):
    """Substitution and scale effects with Cobb-Douglas production.

    Production:     Y = L^α · K^(1-α)  (identical competitive firms)
    Market demand:  P = P̄ - δ·Y   (downward-sloping)
    Input prices:   wage w, rental rate r

    With CRS and perfect competition, industry supply is horizontal at
    the unit cost c(w,r). Equilibrium output is where supply meets
    market demand: P̄ - δY = c(w,r).
    """

    @property
    def name(self) -> str:
        return "Substitution & Scale Effects in Labour Demand"

    @property
    def description(self) -> str:
        return (
            "When wages rise, firms use less labour for two distinct reasons. "
            "The **substitution effect**: labour is now relatively more expensive "
            "than capital, so firms switch toward a more capital-intensive "
            "input mix (moving along the isoquant). The **scale effect**: higher "
            "wages raise the industry's unit cost, shifting supply up along the "
            "market demand curve, which reduces equilibrium output (moving to a "
            "lower isoquant). Both effects reduce labour demand."
        )

    def params(self) -> list[SliderSpec]:
        return [
            SliderSpec("alpha", "Labour share (α)", 0.1, 0.9, 0.5, 0.05,
                       "Exponent on labour in Y = L^α · K^(1-α)"),
            SliderSpec("w_0", "Initial wage (w₀)", 1.0, 20.0, 5.0, 0.5,
                       "Starting wage rate"),
            SliderSpec("w_1", "New wage (w₁)", 1.0, 20.0, 8.0, 0.5,
                       "Wage after the change"),
            SliderSpec("r", "Rental rate of capital (r)", 1.0, 20.0, 5.0, 0.5,
                       "Cost per unit of capital"),
            SliderSpec("P_bar", "Demand intercept (P̄)", 10.0, 100.0, 50.0, 5.0,
                       "Maximum willingness to pay (output demand intercept)"),
            SliderSpec("delta", "Demand slope (δ)", 0.1, 5.0, 1.0, 0.1,
                       "How steeply output demand falls with quantity"),
        ]

    def render(self, params: dict[str, Any], depth: str = "undergraduate") -> None:
        pass

    def _unit_cost(self, w: float, r: float, alpha: float) -> float:
        """Unit cost for Cobb-Douglas: c(w,r) = (w/α)^α · (r/(1-α))^(1-α)."""
        return (w / alpha) ** alpha * (r / (1 - alpha)) ** (1 - alpha)

    def _optimal_output(self, w: float, r: float, alpha: float,
                        P_bar: float, delta: float) -> float:
        """Competitive industry equilibrium: P = MC → P̄ - δY = c(w,r)."""
        c = self._unit_cost(w, r, alpha)
        Y = (P_bar - c) / delta
        return max(Y, 0.0)

    def _conditional_inputs(self, Y: float, w: float, r: float,
                            alpha: float) -> tuple[float, float]:
        """Cost-minimising L, K for given output Y.

        From FOC: K/L = w(1-α)/(rα)
        Sub into Y = L^α · K^(1-α):
            L = Y · (rα / (w(1-α)))^(1-α)
            K = Y · (w(1-α) / (rα))^α
        """
        ratio = r * alpha / (w * (1 - alpha))
        L = Y * ratio ** (1 - alpha)
        K = Y * (1 / ratio) ** alpha
        return L, K

    def compute_curves(self, params: dict[str, Any]) -> dict[str, Any]:
        alpha = params["alpha"]

        # Isoquant: Y = L^α · K^(1-α) → K = (Y / L^α)^(1/(1-α))
        # Return isoquant curves for given Y values
        return {"alpha": alpha}

    def compute_equilibrium(self, params: dict[str, Any]) -> dict[str, Any]:
        alpha = params["alpha"]
        w_0, w_1 = params["w_0"], params["w_1"]
        r = params["r"]
        P_bar, delta = params["P_bar"], params["delta"]

        # 1. Initial equilibrium at w₀
        Y_0 = self._optimal_output(w_0, r, alpha, P_bar, delta)
        L_0, K_0 = self._conditional_inputs(Y_0, w_0, r, alpha)

        # 2. Substitution: new wage w₁, but hold output at Y₀
        L_sub, K_sub = self._conditional_inputs(Y_0, w_1, r, alpha)

        # 3. Final equilibrium at w₁ (scale adjusts)
        Y_1 = self._optimal_output(w_1, r, alpha, P_bar, delta)
        L_1, K_1 = self._conditional_inputs(Y_1, w_1, r, alpha)

        # Decomposition
        substitution_effect = L_sub - L_0
        scale_effect = L_1 - L_sub
        total_effect = L_1 - L_0

        # Unit costs
        c_0 = self._unit_cost(w_0, r, alpha)
        c_1 = self._unit_cost(w_1, r, alpha)

        # Output prices
        P_0 = P_bar - delta * Y_0
        P_1 = P_bar - delta * Y_1

        return {
            # Initial
            "Y_0": Y_0, "L_0": L_0, "K_0": K_0, "c_0": c_0, "P_0": P_0,
            # Substitution (on original isoquant)
            "L_sub": L_sub, "K_sub": K_sub,
            # Final
            "Y_1": Y_1, "L_1": L_1, "K_1": K_1, "c_1": c_1, "P_1": P_1,
            # Decomposition
            "substitution_effect": substitution_effect,
            "scale_effect": scale_effect,
            "total_effect": total_effect,
        }

    def educational_sections(self) -> list:
        sections = [
            ("Substitution effect",
             "When wages rise, labour becomes **relatively more expensive** "
             "compared to capital. To produce the same output at minimum cost, "
             "the firm tilts its input mix toward capital and away from labour. "
             "On the isoquant diagram, this is a movement *along* the original "
             "isoquant to a steeper point — the isocost line has rotated.\n\n"
             "The substitution effect is always negative: a higher wage always "
             "reduces labour demand holding output constant."),
            ("Scale effect",
             "Higher wages raise the industry's unit cost of production, "
             "shifting the (horizontal) supply curve upward. With a "
             "downward-sloping market demand curve, the new equilibrium has a "
             "higher price and lower quantity. The industry **scales back** "
             "production, moving to a lower isoquant and using less of *both* "
             "inputs.\n\n"
             "The scale effect reinforces the substitution effect — both reduce "
             "labour demand when wages rise. This is why the demand curve for "
             "labour always slopes downward."),
            ("Why both effects reduce labour demand",
             "Unlike labour *supply* (where substitution and income effects "
             "work in opposite directions), for labour *demand* both effects "
             "push the same way:\n\n"
             "- **Substitution**: replace expensive labour with cheaper capital\n"
             "- **Scale**: produce less because costs went up\n\n"
             "This is why the labour demand curve is unambiguously downward-"
             "sloping, even though the labour supply curve can bend backward."),
        ]
        sections.append((
                "Shephard's lemma and the cost function (Advanced)",
                "With Cobb-Douglas Y = L^α K^(1-α), the unit cost function is:\n\n"
                r"$$c(w,r) = \left(\frac{w}{\alpha}\right)^\alpha "
                r"\left(\frac{r}{1-\alpha}\right)^{1-\alpha}$$"
                "\n\nBy Shephard's lemma, the conditional demand for labour is:\n\n"
                r"$$L(w,r,Y) = \frac{\partial\, C(w,r,Y)}{\partial w} "
                r"= Y \cdot \alpha \cdot \frac{c(w,r)}{w}$$"
                "\n\nThe substitution effect is the change in L holding Y constant. "
                "The scale effect comes from the change in Y* when costs shift the "
                "profit-maximising output level."
            ))
        return sections


# ---------------------------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------------------------

def main():
    concept = LabourDemand()

    # --- Sidebar ---
    st.sidebar.header("Labour Demand Decomposition")
    st.sidebar.markdown("---")

    st.sidebar.subheader("Production")
    alpha = st.sidebar.slider("Labour share (α)", 0.1, 0.9, 0.5, 0.05,
                              help="Y = L^α · K^(1-α)")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Wage change")
    w_0 = st.sidebar.slider("Initial wage (w₀)", 1.0, 20.0, 5.0, 0.5)
    w_1 = st.sidebar.slider("New wage (w₁)", 1.0, 20.0, 8.0, 0.5)
    r = st.sidebar.slider("Rental rate of capital (r)", 1.0, 20.0, 5.0, 0.5)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Output demand")
    P_bar = st.sidebar.slider("Demand intercept (P̄)", 10.0, 100.0, 50.0, 5.0)
    delta = st.sidebar.slider("Demand slope (δ)", 0.1, 5.0, 1.0, 0.1)

    p = {"alpha": alpha, "w_0": w_0, "w_1": w_1, "r": r,
         "P_bar": P_bar, "delta": delta}
    eq = concept.compute_equilibrium(p)

    # --- Main area ---
    st.title("Substitution & Scale Effects in Labour Demand")
    st.markdown(concept.description)

    st.markdown("---")

    wage_dir = "increase" if w_1 > w_0 else "decrease" if w_1 < w_0 else "change"
    st.markdown(f"""
**How this app works:** A competitive industry produces output using labour
and capital with a Cobb-Douglas production function (Y = L^α · K^(1-α))
and faces a downward-sloping market demand curve. With constant returns to
scale, industry supply is horizontal at the unit cost. The sidebar sets the
initial wage, a new wage, and the capital rental rate. The app decomposes
the effect of the wage {wage_dir} on industry labour demand into a
substitution effect (re-optimising the input mix at constant output) and a
scale effect (the output change as supply shifts along the demand curve).
""")

    st.markdown("---")

    # Check valid equilibrium
    if eq["Y_0"] <= 0:
        st.error("No production at the initial wage — unit cost exceeds the "
                 "demand intercept. Lower wages or raise P̄.")
        return
    if eq["Y_1"] <= 0:
        st.warning("The firm shuts down at the new wage — costs are too high.")

    # --- Plots ---

    # 1. Isoquant / Isocost (full width)
    st.markdown("**Isoquant / Isocost Diagram**")
    fig1, ax1 = plt.subplots(figsize=(9, 5.5))

    L_max = max(eq["L_0"], eq["L_sub"], eq["L_1"]) * 2.5
    L_range = np.linspace(0.01, L_max, 500)

    K_iso0 = (eq["Y_0"] / L_range ** alpha) ** (1 / (1 - alpha))
    ax1.plot(L_range, K_iso0, color="#2563eb", linewidth=2,
             label=f"Y₀ = {eq['Y_0']:.1f}")

    if eq["Y_1"] > 0:
        K_iso1 = (eq["Y_1"] / L_range ** alpha) ** (1 / (1 - alpha))
        ax1.plot(L_range, K_iso1, color="#2563eb", linewidth=1.5,
                 linestyle="--", label=f"Y₁ = {eq['Y_1']:.1f}")

    C_0 = w_0 * eq["L_0"] + r * eq["K_0"]
    C_sub = w_1 * eq["L_sub"] + r * eq["K_sub"]
    C_1 = w_1 * eq["L_1"] + r * eq["K_1"]
    K_max_plot = max(eq["K_0"], eq["K_sub"], eq["K_1"]) * 2.5
    L_line = np.linspace(0, L_max, 200)

    K_ic0 = (C_0 - w_0 * L_line) / r
    ax1.plot(L_line[K_ic0 >= 0], K_ic0[K_ic0 >= 0],
             color="#dc2626", linewidth=1.2, alpha=0.6,
             label=f"Isocost w₀={w_0:.1f}")

    K_ic_sub = (C_sub - w_1 * L_line) / r
    ax1.plot(L_line[K_ic_sub >= 0], K_ic_sub[K_ic_sub >= 0],
             color="#f59e0b", linewidth=1.2, alpha=0.6,
             label=f"Isocost w₁={w_1:.1f} (Y₀)")

    if eq["Y_1"] > 0:
        K_ic1 = (C_1 - w_1 * L_line) / r
        ax1.plot(L_line[K_ic1 >= 0], K_ic1[K_ic1 >= 0],
                 color="#f59e0b", linewidth=1.2, linestyle="--", alpha=0.6,
                 label=f"Isocost w₁={w_1:.1f} (Y₁)")

    ax1.plot(eq["L_0"], eq["K_0"], "o", color="#dc2626", markersize=11,
             zorder=6, label="A (initial)")
    ax1.plot(eq["L_sub"], eq["K_sub"], "s", color="#f59e0b", markersize=11,
             zorder=6, label="B (substitution)")
    if eq["Y_1"] > 0:
        ax1.plot(eq["L_1"], eq["K_1"], "^", color="#16a34a", markersize=11,
                 zorder=6, label="C (final)")

    ax1.annotate("", xy=(eq["L_sub"], eq["K_sub"]),
                 xytext=(eq["L_0"], eq["K_0"]),
                 arrowprops=dict(arrowstyle="->", color="#7c3aed", lw=2.5))
    if eq["Y_1"] > 0:
        ax1.annotate("", xy=(eq["L_1"], eq["K_1"]),
                     xytext=(eq["L_sub"], eq["K_sub"]),
                     arrowprops=dict(arrowstyle="->", color="#16a34a", lw=2.5))

    ax1.set_xlabel("Labour (L)", fontsize=11)
    ax1.set_ylabel("Capital (K)", fontsize=11)
    ax1.set_xlim(0, L_max)
    ax1.set_ylim(0, K_max_plot)
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(True, alpha=0.2)
    ax1.set_title("A → B: Substitution (along isoquant)    "
                  "B → C: Scale (to lower isoquant)",
                  fontsize=11, fontweight="bold")
    fig1.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)

    # 2–3. Output market + decomposition bar chart (side by side)
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**Output Market (Scale Effect)**")
        fig2, ax2 = plt.subplots(figsize=(6, 4.5))

        Y_max = eq["Y_0"] * 1.8
        Y_range = np.linspace(0, Y_max, 200)

        P_demand = P_bar - delta * Y_range
        ax2.plot(Y_range, P_demand, color="#2563eb", linewidth=2,
                 label="Market demand")

        ax2.axhline(eq["c_0"], color="#dc2626", linewidth=2,
                    label=f"Supply₀ (c = {eq['c_0']:.2f})")
        ax2.axhline(eq["c_1"], color="#16a34a", linewidth=2, linestyle="--",
                    label=f"Supply₁ (c = {eq['c_1']:.2f})")

        ax2.plot(eq["Y_0"], eq["c_0"], "o", color="#dc2626", markersize=9, zorder=6)
        if eq["Y_1"] > 0:
            ax2.plot(eq["Y_1"], eq["c_1"], "^", color="#16a34a", markersize=9, zorder=6)

        ax2.vlines(eq["Y_0"], 0, eq["c_0"], colors="#dc2626",
                   linestyles=":", linewidth=1, alpha=0.5)
        if eq["Y_1"] > 0:
            ax2.vlines(eq["Y_1"], 0, eq["c_1"], colors="#16a34a",
                       linestyles=":", linewidth=1, alpha=0.5)

        if eq["Y_1"] > 0 and abs(eq["Y_0"] - eq["Y_1"]) > 0.1:
            mid_mc = (eq["c_0"] + eq["c_1"]) / 2
            ax2.annotate("", xy=(eq["Y_1"], mid_mc), xytext=(eq["Y_0"], mid_mc),
                         arrowprops=dict(arrowstyle="->", color="#7c3aed", lw=2.5))
            ax2.text((eq["Y_0"] + eq["Y_1"]) / 2, mid_mc + 1, "ΔY",
                     color="#7c3aed", fontsize=10, fontweight="bold",
                     ha="center", va="bottom")

        ax2.text(eq["Y_0"] + 0.3, eq["c_0"] + 0.5, f"Y₀={eq['Y_0']:.1f}",
                 fontsize=9, color="#dc2626")
        if eq["Y_1"] > 0:
            ax2.text(eq["Y_1"] + 0.3, eq["c_1"] + 0.5, f"Y₁={eq['Y_1']:.1f}",
                     fontsize=9, color="#16a34a")

        ax2.set_xlabel("Industry output (Y)", fontsize=10)
        ax2.set_ylabel("Price / Unit cost ($)", fontsize=10)
        ax2.set_xlim(0, Y_max)
        ax2.set_ylim(0, P_bar * 1.05)
        ax2.legend(fontsize=8, loc="upper right")
        ax2.grid(True, alpha=0.2)
        ax2.set_title("w↑ → cost↑ → supply↑ → Y↓",
                      fontsize=12, fontweight="bold")
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    with col_right:
        st.markdown("**Decomposition of ΔL**")
        fig3, ax3 = plt.subplots(figsize=(6, 4.5))

        labels = ["Total", "Substitution", "Scale"]
        values = [eq["total_effect"], eq["substitution_effect"], eq["scale_effect"]]
        colors = ["#64748b", "#7c3aed", "#16a34a"]

        bars = ax3.bar(labels, values, color=colors, edgecolor="white", width=0.5)
        for bar, val in zip(bars, values):
            y = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2, y,
                     f"{val:+.2f}", ha="center",
                     va="bottom" if y >= 0 else "top",
                     fontsize=11, fontweight="bold")

        ax3.axhline(0, color="black", linewidth=0.5)
        ax3.set_ylabel("Change in labour (ΔL)", fontsize=10)
        ax3.grid(True, alpha=0.2, axis="y")
        ax3.set_title("Decomposition", fontsize=12, fontweight="bold")
        fig3.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

    st.caption(
        "**Top:** A → B is the substitution effect (along the Y₀ isoquant — "
        "same output, new input mix). B → C is the scale effect (to a lower "
        "isoquant). "
        "**Bottom left:** The output market shows *why* the scale effect happens "
        "— higher wages raise the industry's unit cost, shifting supply up "
        "along the market demand curve, so less is produced. "
        "**Bottom right:** The bar chart decomposes the total change in labour."
    )

    st.markdown("---")

    # --- Metrics ---
    st.markdown("**Equilibrium Values**")
    c1, c2, c3 = st.columns(3)
    c1.markdown("*Initial (A)*")
    c1.text(f"  L = {eq['L_0']:.2f}\n  K = {eq['K_0']:.2f}\n  Y = {eq['Y_0']:.2f}")
    c2.markdown("*Substitution (B)*")
    c2.text(f"  L = {eq['L_sub']:.2f}\n  K = {eq['K_sub']:.2f}\n  Y = {eq['Y_0']:.2f}")
    c3.markdown("*Final (C)*")
    if eq["Y_1"] > 0:
        c3.text(f"  L = {eq['L_1']:.2f}\n  K = {eq['K_1']:.2f}\n  Y = {eq['Y_1']:.2f}")
    else:
        c3.text("  Firm shuts down")

    st.markdown("---")
    st.markdown("**Decomposition**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Substitution effect", f"{eq['substitution_effect']:+.2f}")
    c2.metric("Scale effect", f"{eq['scale_effect']:+.2f}")
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
        page_title="Labour Demand Decomposition",
        page_icon="📊",
        layout="wide",
    )
    main()
