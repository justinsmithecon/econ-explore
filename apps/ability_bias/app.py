"""Ability Bias in Returns to Schooling — Streamlit app.

Show how omitting ability from a wage regression biases the estimated
return to schooling upward when ability is positively correlated with
both schooling and wages.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import streamlit as st

from shared.base import SliderSpec, EstimationConcept


class AbilityBias(EstimationConcept):
    """Ability bias as an omitted-variable-bias problem.

    True DGP:
        wage = β₀ + β₁·schooling + β₂·ability + ε

    Ability is unobserved, so the naive regression omits it:
        wage = α₀ + α₁·schooling + u

    OVB formula:
        plim α̂₁ = β₁ + β₂ · δ₁

    where δ₁ = Cov(S, A) / Var(S) = ρ · σ_A / σ_S is the coefficient
    from regressing ability on schooling.

    Since β₂ > 0 (ability raises wages) and δ₁ > 0 (more able people
    get more schooling), the naive estimate is upward-biased.
    """

    @property
    def name(self) -> str:
        return "Ability Bias in Returns to Schooling"

    @property
    def description(self) -> str:
        return (
            "A classic problem in labour economics: people with higher "
            "innate ability tend to get more schooling *and* earn higher "
            "wages. If ability is unobserved, a simple regression of wages "
            "on schooling conflates the true return to education with the "
            "wage premium from ability — **biasing the estimated return to "
            "schooling upward**. This is a textbook case of omitted "
            "variable bias (OVB)."
        )

    def params(self) -> list[SliderSpec]:
        return [
            SliderSpec("beta_1", "True return to schooling (β₁)", 0.0, 5.0, 1.5, 0.1,
                       "Extra $/hr per year of schooling, holding ability constant"),
            SliderSpec("beta_2", "Return to ability (β₂)", 0.0, 5.0, 2.0, 0.1,
                       "Extra $/hr per unit of ability, holding schooling constant"),
            SliderSpec("rho", "Corr(schooling, ability) (ρ)", -0.5, 0.95, 0.6, 0.05,
                       "How strongly ability predicts schooling"),
            SliderSpec("sigma", "Wage noise (σ)", 0.5, 10.0, 3.0, 0.5,
                       "Standard deviation of idiosyncratic wage shocks"),
            SliderSpec("n", "Sample size", 100.0, 3000.0, 500.0, 100.0,
                       "Number of workers in the sample"),
        ]

    def render(self, params: dict[str, Any], depth: str = "undergraduate") -> None:
        pass

    def generate_data(self, params: dict[str, Any],
                      rng: np.random.Generator) -> dict[str, Any]:
        n = int(params["n"])
        rho = params["rho"]
        beta_0, beta_1, beta_2 = 5.0, params["beta_1"], params["beta_2"]
        sigma = params["sigma"]

        # Schooling and ability as bivariate normal
        # schooling ~ N(14, 2²), ability ~ N(0, 1), cor = ρ
        z1 = rng.standard_normal(n)
        z2 = rng.standard_normal(n)

        sd_s = 2.0
        mean_s = 14.0

        schooling = mean_s + sd_s * z1
        ability = rho * z1 + np.sqrt(max(1 - rho ** 2, 0)) * z2

        wage = beta_0 + beta_1 * schooling + beta_2 * ability + rng.normal(0, sigma, n)

        return {
            "schooling": schooling,
            "ability": ability,
            "wage": wage,
            "beta_0": beta_0,
        }

    def estimate(self, data: dict[str, Any],
                 params: dict[str, Any]) -> dict[str, Any]:
        schooling = data["schooling"]
        ability = data["ability"]
        wage = data["wage"]
        n = len(wage)
        ones = np.ones(n)

        # Long regression: wage ~ schooling + ability (the truth)
        X_long = np.column_stack([ones, schooling, ability])
        beta_long, _, _, _ = np.linalg.lstsq(X_long, wage, rcond=None)

        # Short regression: wage ~ schooling (naive, omitting ability)
        X_short = np.column_stack([ones, schooling])
        beta_short, _, _, _ = np.linalg.lstsq(X_short, wage, rcond=None)

        # Auxiliary regression: ability ~ schooling (measures the correlation)
        delta, _, _, _ = np.linalg.lstsq(X_short, ability, rcond=None)

        # Analytic bias: β₂ · δ₁
        bias_estimated = beta_short[1] - beta_long[1]
        bias_formula = beta_long[2] * delta[1]

        # Theoretical OVB (from population parameters)
        rho = params["rho"]
        sd_s = 2.0
        # δ₁ = ρ · σ_A / σ_S = ρ / sd_s (since σ_A = 1)
        delta_1_pop = rho / sd_s
        bias_theoretical = params["beta_2"] * delta_1_pop

        return {
            "beta_long": beta_long,      # [β₀, β₁, β₂]
            "beta_short": beta_short,     # [α₀, α₁]
            "delta": delta,               # [δ₀, δ₁]
            "bias_estimated": bias_estimated,
            "bias_formula": bias_formula,
            "bias_theoretical": bias_theoretical,
            "delta_1_pop": delta_1_pop,
        }

    def educational_sections(self, depth: str = "undergraduate") -> list:
        sections = [
            ("What is ability bias?",
             "The **Mincer earnings equation** regresses log wages on years "
             "of schooling to estimate the 'return to education'. But people "
             "don't choose schooling at random — those with higher ability "
             "(intelligence, motivation, family resources) tend to get more "
             "schooling.\n\n"
             "If ability also directly raises wages (beyond its effect through "
             "schooling), then a simple regression attributes *some of ability's "
             "effect* to schooling. This is **ability bias** — a specific case "
             "of omitted variable bias."),
            ("The OVB formula",
             "With the true model wage = β₀ + β₁·S + β₂·A + ε, if we omit A "
             "and run wage = α₀ + α₁·S + u, then:\n\n"
             r"$$\text{plim}\;\hat{\alpha}_1 = \beta_1 + \beta_2 \cdot \delta_1$$"
             "\n\nwhere δ₁ = Cov(S, A) / Var(S) is the slope from regressing "
             "ability on schooling.\n\n"
             "**The bias is β₂ · δ₁.** It is positive when:\n"
             "- β₂ > 0 (ability raises wages) *and* δ₁ > 0 (ability raises "
             "schooling)\n\n"
             "So the naive return to schooling is **too high** — it picks up "
             "part of the ability premium."),
            ("Why this matters for policy",
             "If the true return to schooling is β₁ = $1.50/hr but we estimate "
             "$2.50/hr because of ability bias, we'd overstate the benefit of "
             "policies that increase educational attainment.\n\n"
             "Solutions used in the literature:\n"
             "- **Instrumental variables**: use exogenous variation in schooling "
             "(e.g. compulsory schooling laws, distance to college)\n"
             "- **Twin studies**: compare twins with different schooling levels "
             "(same genes/family)\n"
             "- **Natural experiments**: exploit policy changes that affected "
             "schooling but not ability"),
            ("Try it yourself",
             "Use the sliders to explore:\n"
             "- **Set ρ = 0**: no correlation between ability and schooling → "
             "no bias. The naive and true estimates coincide.\n"
             "- **Increase ρ**: ability and schooling become more correlated → "
             "larger upward bias.\n"
             "- **Set β₂ = 0**: ability doesn't affect wages directly → no bias "
             "even if ability and schooling are correlated.\n"
             "- **Increase n**: sampling variance shrinks, estimates converge "
             "to their probability limits, and the bias becomes more visible."),
        ]
        if depth == "graduate":
            sections.append((
                "Formal derivation",
                "Write the DGP in matrix form: **y** = **X**β + ε where "
                "**X** = [**1**, **S**, **A**]. The short regression omits "
                "**A**, so:\n\n"
                r"$$\hat{\alpha}_1 = (\mathbf{S}'\mathbf{M}_1\mathbf{S})^{-1}"
                r"\mathbf{S}'\mathbf{M}_1\mathbf{y}$$"
                "\n\nwhere **M**₁ demeans. Substituting the true DGP:\n\n"
                r"$$\hat{\alpha}_1 = \beta_1 + \beta_2 \cdot "
                r"\frac{\mathbf{S}'\mathbf{M}_1\mathbf{A}}"
                r"{\mathbf{S}'\mathbf{M}_1\mathbf{S}} + "
                r"\frac{\mathbf{S}'\mathbf{M}_1\boldsymbol{\varepsilon}}"
                r"{\mathbf{S}'\mathbf{M}_1\mathbf{S}}$$"
                "\n\nThe second fraction vanishes as n → ∞ (by the LLN). "
                "The first fraction converges to δ₁ = Cov(S,A)/Var(S). "
                "Hence plim α̂₁ = β₁ + β₂δ₁.\n\n"
                "The sign and magnitude of the bias depend entirely on β₂ "
                "(the direct effect of the omitted variable) and δ₁ (the "
                "relationship between the omitted and included variables)."
            ))
        return sections


# ---------------------------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------------------------

def main():
    concept = AbilityBias()

    # --- Sidebar ---
    st.sidebar.header("Ability Bias")
    st.sidebar.markdown("---")

    st.sidebar.subheader("True DGP")
    beta_1 = st.sidebar.slider(
        "True return to schooling (β₁)", 0.0, 5.0, 1.5, 0.1,
        help="$/hr per year of schooling, holding ability constant")
    beta_2 = st.sidebar.slider(
        "Return to ability (β₂)", 0.0, 5.0, 2.0, 0.1,
        help="$/hr per unit of ability, holding schooling constant")
    rho = st.sidebar.slider(
        "Corr(schooling, ability) (ρ)", -0.5, 0.95, 0.6, 0.05,
        help="Correlation between innate ability and years of schooling")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Sample")
    n = st.sidebar.slider("Sample size", 100, 3000, 500, 100)
    sigma = st.sidebar.slider("Wage noise (σ)", 0.5, 10.0, 3.0, 0.5)

    st.sidebar.markdown("---")
    depth = st.sidebar.radio("Depth", ["undergraduate", "graduate"],
                             format_func=lambda x: x.title())
    use_seed = st.sidebar.checkbox("Fix random seed", value=True)
    seed = 42 if use_seed else None

    p = {"beta_1": beta_1, "beta_2": beta_2, "rho": rho,
         "sigma": sigma, "n": float(n)}

    rng = np.random.default_rng(seed)
    data = concept.generate_data(p, rng)
    result = concept.estimate(data, p)

    # --- Main area ---
    st.title("Ability Bias in Returns to Schooling")
    st.markdown(concept.description)

    st.markdown("---")

    bias_dir = "upward" if rho > 0 and beta_2 > 0 else (
        "downward" if rho * beta_2 < 0 else "no")
    st.markdown(f"""
**How this app works:** The true data-generating process is
wage = β₀ + β₁·schooling + β₂·ability + ε, where ability and schooling
are correlated with strength ρ. Since ability is unobserved, the naive
regression omits it. The sidebar sets the true parameters and the
correlation. The app draws a sample, runs both the correct (long) and
naive (short) regressions, and shows the resulting {bias_dir} bias
in the estimated return to schooling. The OVB formula
(bias = β₂ · δ₁) is verified against the sample estimates.
""")

    st.markdown("---")

    # --- Headline metrics ---
    st.markdown("**Estimated Return to Schooling**")
    c1, c2, c3 = st.columns(3)
    c1.metric("True β₁ (parameter)", f"${beta_1:.2f}/hr")
    c2.metric("Long regression (controls for ability)",
              f"${result['beta_long'][1]:.2f}/hr")
    c3.metric("Naive regression (omits ability)",
              f"${result['beta_short'][1]:.2f}/hr",
              delta=f"{result['bias_estimated']:+.2f} bias",
              delta_color="inverse")

    # --- Plots ---

    # 1. Wage vs Schooling scatter with both regression lines (full width)
    st.markdown("**Wage vs Schooling — Naive and Controlled Estimates**")
    fig1, ax1 = plt.subplots(figsize=(9, 5.5))

    # Color points by ability
    ability = data["ability"]
    norm = mcolors.Normalize(vmin=np.percentile(ability, 5),
                             vmax=np.percentile(ability, 95))
    sc = ax1.scatter(data["schooling"], data["wage"], c=ability,
                     cmap="RdYlBu_r", norm=norm,
                     alpha=0.4, s=12, edgecolors="none", rasterized=True)
    cbar = fig1.colorbar(sc, ax=ax1, shrink=0.8, pad=0.02)
    cbar.set_label("Ability (unobserved)", fontsize=10)

    # Regression lines
    s_range = np.linspace(data["schooling"].min() - 0.5,
                          data["schooling"].max() + 0.5, 200)

    # Naive (short) regression line
    line_short = result["beta_short"][0] + result["beta_short"][1] * s_range
    ax1.plot(s_range, line_short, color="#dc2626", linewidth=2.5,
             label=f"Naive: slope = {result['beta_short'][1]:.2f}")

    # Long regression line (at mean ability = 0)
    line_long = (result["beta_long"][0]
                 + result["beta_long"][1] * s_range
                 + result["beta_long"][2] * 0)  # ability at mean
    ax1.plot(s_range, line_long, color="#2563eb", linewidth=2.5,
             linestyle="--",
             label=f"Controlled: slope = {result['beta_long'][1]:.2f}")

    # Annotate the gap between slopes
    s_annot = data["schooling"].mean() + 3
    y_short = result["beta_short"][0] + result["beta_short"][1] * s_annot
    y_long = (result["beta_long"][0] + result["beta_long"][1] * s_annot)
    if abs(y_short - y_long) > 0.3:
        ax1.annotate(
            "", xy=(s_annot, y_short), xytext=(s_annot, y_long),
            arrowprops=dict(arrowstyle="<->", color="#7c3aed", lw=2.5))
        ax1.text(s_annot + 0.2, (y_short + y_long) / 2, "Bias",
                 color="#7c3aed", fontsize=11, fontweight="bold",
                 va="center", ha="left",
                 bbox=dict(facecolor="white", edgecolor="#7c3aed",
                           alpha=0.9, boxstyle="round,pad=0.2"))

    ax1.set_xlabel("Years of schooling", fontsize=11)
    ax1.set_ylabel("Wage ($/hr)", fontsize=11)
    ax1.legend(fontsize=9, loc="upper left")
    ax1.grid(True, alpha=0.2)
    ax1.set_title(
        "Red (naive) line is steeper — it conflates schooling and ability"
        if result["bias_estimated"] > 0.05
        else "Lines overlap — little or no ability bias",
        fontsize=11, fontweight="bold")
    fig1.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)

    # 2–3. Coefficient comparison + ability-schooling scatter (side by side)
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**OVB Decomposition**")
        fig2, ax2 = plt.subplots(figsize=(6, 4.5))

        labels = ["True β₁", "Bias\n(β₂ · δ₁)", "Naive α̂₁"]
        true_val = result["beta_long"][1]
        bias_val = result["bias_estimated"]
        naive_val = result["beta_short"][1]

        # Stacked bar: true + bias = naive
        ax2.bar(["Naive\nestimate"], [true_val], color="#2563eb",
                edgecolor="white", width=0.4, label=f"True effect (β̂₁ = {true_val:.2f})")
        ax2.bar(["Naive\nestimate"], [bias_val], bottom=[true_val],
                color="#dc2626", edgecolor="white", width=0.4,
                label=f"Bias (β̂₂·δ̂₁ = {bias_val:.2f})")

        # Reference lines
        ax2.axhline(naive_val, color="#dc2626", linewidth=1, linestyle=":",
                     alpha=0.5)
        ax2.axhline(true_val, color="#2563eb", linewidth=1, linestyle=":",
                     alpha=0.5)

        # Value labels
        ax2.text(0, true_val / 2, f"${true_val:.2f}",
                 ha="center", va="center", fontsize=11,
                 fontweight="bold", color="white")
        if abs(bias_val) > 0.1:
            ax2.text(0, true_val + bias_val / 2, f"${bias_val:.2f}",
                     ha="center", va="center", fontsize=11,
                     fontweight="bold", color="white")
        ax2.text(0.25, naive_val, f"  α̂₁ = ${naive_val:.2f}",
                 ha="left", va="bottom", fontsize=9, color="#dc2626")
        ax2.text(0.25, true_val, f"  β̂₁ = ${true_val:.2f}",
                 ha="left", va="top", fontsize=9, color="#2563eb")

        ax2.set_ylabel("$/hr per year of schooling", fontsize=10)
        ax2.legend(fontsize=8, loc="upper right")
        ax2.grid(True, alpha=0.2, axis="y")
        ax2.set_xlim(-0.5, 0.8)
        ax2.set_title("Naive = True + Bias", fontsize=12, fontweight="bold")
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    with col_right:
        st.markdown("**Schooling vs Ability (the source of bias)**")
        fig3, ax3 = plt.subplots(figsize=(6, 4.5))

        ax3.scatter(data["schooling"], data["ability"],
                    alpha=0.3, s=10, color="#64748b", edgecolors="none",
                    rasterized=True)

        # Auxiliary regression line: ability = δ₀ + δ₁·schooling
        ax3.plot(s_range,
                 result["delta"][0] + result["delta"][1] * s_range,
                 color="#7c3aed", linewidth=2.5,
                 label=f"δ̂₁ = {result['delta'][1]:.3f}")

        ax3.set_xlabel("Years of schooling", fontsize=10)
        ax3.set_ylabel("Ability (unobserved)", fontsize=10)
        ax3.legend(fontsize=9, loc="upper left")
        ax3.grid(True, alpha=0.2)
        ax3.set_title(f"ρ = {rho:.2f} — higher ability → more schooling"
                      if rho > 0
                      else f"ρ = {rho:.2f}" if rho == 0
                      else f"ρ = {rho:.2f} — higher ability → less schooling",
                      fontsize=11, fontweight="bold")
        fig3.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

    st.caption(
        "**Top:** Wage vs schooling, with points coloured by ability. The "
        "naive (red) line is steeper than the controlled (blue) line because "
        "it picks up the wage premium from ability. "
        "**Bottom left:** The stacked bar decomposes the naive estimate into "
        "the true return and the bias. "
        "**Bottom right:** The auxiliary regression shows *why* the bias "
        "exists — ability and schooling are correlated (δ₁ > 0)."
    )

    st.markdown("---")

    # --- OVB formula verification ---
    st.markdown("**OVB Formula Verification**")
    c1, c2, c3 = st.columns(3)
    c1.markdown("*Components*")
    c1.text(f"  β̂₂ (ability effect) = {result['beta_long'][2]:.3f}\n"
            f"  δ̂₁ (aux. slope)     = {result['delta'][1]:.3f}\n"
            f"  β̂₂ × δ̂₁            = {result['bias_formula']:.3f}")
    c2.markdown("*Actual bias*")
    c2.text(f"  α̂₁ − β̂₁            = {result['bias_estimated']:.3f}")
    c3.markdown("*Theoretical (population)*")
    c3.text(f"  β₂ × δ₁             = {result['bias_theoretical']:.3f}\n"
            f"  ({beta_2:.2f} × {result['delta_1_pop']:.3f})")

    # --- All coefficients ---
    st.markdown("---")
    st.markdown("**All Estimated Coefficients**")
    c1, c2, c3 = st.columns(3)
    c1.markdown("*Long regression* (wage ~ S + A)")
    c1.text(f"  Intercept  = {result['beta_long'][0]:.3f}  (true: 5.000)\n"
            f"  Schooling  = {result['beta_long'][1]:.3f}  (true: {beta_1:.3f})\n"
            f"  Ability    = {result['beta_long'][2]:.3f}  (true: {beta_2:.3f})")
    c2.markdown("*Short regression* (wage ~ S)")
    c2.text(f"  Intercept  = {result['beta_short'][0]:.3f}\n"
            f"  Schooling  = {result['beta_short'][1]:.3f}  (biased)")
    c3.markdown("*Auxiliary* (A ~ S)")
    c3.text(f"  Intercept  = {result['delta'][0]:.3f}\n"
            f"  Schooling  = {result['delta'][1]:.3f}")

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
        page_title="Ability Bias — Returns to Schooling",
        page_icon="📊",
        layout="wide",
    )
    main()
