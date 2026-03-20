"""Statistical Discrimination — Streamlit app.

Explore the Arrow/Phelps model where employers use group membership
as a proxy for productivity when individual signals are noisy,
leading to rational but unfair wage differences.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from scipy import stats

from shared.base import SliderSpec, EquilibriumConcept


class StatisticalDiscrimination(EquilibriumConcept):
    """Bayesian statistical discrimination with two groups.

    True productivity:  θ ~ N(μ_g, σ²_θ)  for group g ∈ {A, B}
    Noisy signal:       s = θ + ε,  ε ~ N(0, σ²_ε_g)
    Signal precision differs across groups.

    Employer's posterior (Bayesian updating):
        E[θ | s, g] = B_g · μ_g + (1 − B_g) · s

    where B_g = σ²_ε_g / (σ²_θ + σ²_ε_g) is the shrinkage factor.

    High B_g (noisy signal) → wage pulled toward group mean.
    Low B_g (precise signal) → wage tracks individual signal.
    """

    @property
    def name(self) -> str:
        return "Statistical Discrimination"

    @property
    def description(self) -> str:
        return (
            "In the Arrow/Phelps model of **statistical discrimination**, "
            "employers cannot perfectly observe a worker's productivity. "
            "They see a noisy signal and rationally combine it with their "
            "prior belief about the worker's group. When the signal is "
            "noisier for one group, employers rely more on the group average "
            "for that group — **shrinking individual wages toward the group "
            "mean**. This can produce wage gaps even when both groups have "
            "identical true productivity distributions."
        )

    def params(self) -> list[SliderSpec]:
        return [
            SliderSpec("mu_A", "Group A mean productivity (μ_A)", 5.0, 30.0, 15.0, 1.0,
                       "Average true productivity for Group A"),
            SliderSpec("mu_B", "Group B mean productivity (μ_B)", 5.0, 30.0, 15.0, 1.0,
                       "Average true productivity for Group B"),
            SliderSpec("sigma_theta", "Productivity SD (σ_θ)", 1.0, 10.0, 4.0, 0.5,
                       "Standard deviation of true productivity (same for both groups)"),
            SliderSpec("sigma_eps_A", "Signal noise for A (σ_ε_A)", 0.5, 15.0, 2.0, 0.5,
                       "How noisy the productivity signal is for Group A"),
            SliderSpec("sigma_eps_B", "Signal noise for B (σ_ε_B)", 0.5, 15.0, 6.0, 0.5,
                       "How noisy the productivity signal is for Group B"),
        ]

    def render(self, params: dict[str, Any], depth: str = "undergraduate") -> None:
        pass

    def compute_curves(self, params: dict[str, Any]) -> dict[str, Any]:
        return {}

    def compute_equilibrium(self, params: dict[str, Any]) -> dict[str, Any]:
        mu_A = params["mu_A"]
        mu_B = params["mu_B"]
        sigma_theta = params["sigma_theta"]
        sigma_eps_A = params["sigma_eps_A"]
        sigma_eps_B = params["sigma_eps_B"]

        var_theta = sigma_theta ** 2
        var_eps_A = sigma_eps_A ** 2
        var_eps_B = sigma_eps_B ** 2

        # Shrinkage factors: how much wage is pulled toward group mean
        B_A = var_eps_A / (var_theta + var_eps_A)
        B_B = var_eps_B / (var_theta + var_eps_B)

        # Wage function slopes (weight on signal)
        slope_A = 1 - B_A   # = var_theta / (var_theta + var_eps_A)
        slope_B = 1 - B_B

        # Wage function intercepts (weight on prior × prior mean)
        intercept_A = B_A * mu_A
        intercept_B = B_B * mu_B

        # Signal distributions: s ~ N(μ_g, σ²_θ + σ²_ε_g)
        sigma_s_A = np.sqrt(var_theta + var_eps_A)
        sigma_s_B = np.sqrt(var_theta + var_eps_B)

        # Wage offer distributions:
        # w = B·μ + (1-B)·s, where s ~ N(μ, σ²_θ + σ²_ε)
        # So w ~ N(μ, (1-B)²·(σ²_θ + σ²_ε))
        # = N(μ, σ²_θ·(σ²_θ/(σ²_θ+σ²_ε)))  -- variance shrinks
        wage_mean_A = mu_A  # E[w] = B·μ + (1-B)·μ = μ
        wage_mean_B = mu_B
        wage_sd_A = slope_A * sigma_s_A
        wage_sd_B = slope_B * sigma_s_B

        # Average wage gap (= gap in means, regardless of noise)
        avg_wage_gap = mu_A - mu_B

        # Wage gap at a specific signal
        # w_A(s) - w_B(s) = (B_A·μ_A + (1-B_A)·s) - (B_B·μ_B + (1-B_B)·s)
        #                  = B_A·μ_A - B_B·μ_B + (B_B - B_A)·s
        gap_intercept = B_A * mu_A - B_B * mu_B
        gap_slope = B_B - B_A  # how gap changes with signal

        # Signal where gap = 0 (if gap_slope ≠ 0)
        if abs(gap_slope) > 1e-10:
            s_equal = -gap_intercept / gap_slope
        else:
            s_equal = None

        # Posterior variance (how uncertain the employer remains)
        post_var_A = var_theta * var_eps_A / (var_theta + var_eps_A)
        post_var_B = var_theta * var_eps_B / (var_theta + var_eps_B)

        return {
            "B_A": B_A, "B_B": B_B,
            "slope_A": slope_A, "slope_B": slope_B,
            "intercept_A": intercept_A, "intercept_B": intercept_B,
            "sigma_s_A": sigma_s_A, "sigma_s_B": sigma_s_B,
            "wage_mean_A": wage_mean_A, "wage_mean_B": wage_mean_B,
            "wage_sd_A": wage_sd_A, "wage_sd_B": wage_sd_B,
            "avg_wage_gap": avg_wage_gap,
            "gap_intercept": gap_intercept, "gap_slope": gap_slope,
            "s_equal": s_equal,
            "post_var_A": post_var_A, "post_var_B": post_var_B,
        }

    def wage_offer(self, s: float | np.ndarray, group: str,
                   eq: dict[str, Any]) -> float | np.ndarray:
        """Compute the wage offer for signal s and group."""
        if group == "A":
            return eq["intercept_A"] + eq["slope_A"] * s
        return eq["intercept_B"] + eq["slope_B"] * s

    def educational_sections(self) -> list:
        sections = [
            ("What is statistical discrimination?",
             "**Taste-based discrimination** (Becker, 1957) assumes employers "
             "have a preference *against* certain groups. **Statistical "
             "discrimination** (Arrow, 1973; Phelps, 1972) is different — "
             "employers are perfectly rational and harbour no prejudice. They "
             "simply use all available information, including group membership, "
             "to form the best possible estimate of a worker's productivity.\n\n"
             "The problem is that this 'rational' behaviour produces unequal "
             "outcomes: two workers with identical signals but from different "
             "groups receive different wages."),
            ("The role of signal noise",
             "The key parameter is **how noisy the productivity signal is** "
             "for each group. If Group B's signal is noisier (σ_ε_B > σ_ε_A), "
             "then for Group B workers the employer puts more weight on the "
             "group average and less on the individual signal.\n\n"
             "**Why might signals differ across groups?**\n"
             "- Employers may be less familiar with Group B's credentials\n"
             "- Screening tests may be culturally biased\n"
             "- Referral networks may transmit better information for Group A\n"
             "- Fewer observations of Group B workers → less precise inference"),
            ("Who gains and who loses?",
             "Statistical discrimination doesn't just shift the average wage — "
             "it **compresses the wage distribution** for the noisier group:\n\n"
             "- **High-ability workers** in the noisy group are paid less than "
             "equally talented workers in the precise group (their signal is "
             "discounted)\n"
             "- **Low-ability workers** in the noisy group are paid more than "
             "their counterparts (they benefit from regression to the mean)\n\n"
             "The average wage is unaffected if both groups have the same mean "
             "productivity — but individual fairness is violated."),
            ("Policy implications",
             "If discrimination is statistical (not taste-based), the policy "
             "responses differ:\n\n"
             "- **Banning group-based pricing** (e.g. blind CV policies) forces "
             "employers to use a pooled prior — this helps some but can hurt "
             "others\n"
             "- **Improving signal quality** for the disadvantaged group "
             "(better credentialing, standardised testing) directly reduces "
             "discrimination\n"
             "- **Affirmative action** can address the feedback loop: if "
             "hiring more Group B workers generates better information about "
             "them, signals improve over time"),
        ]
        sections.append((
                "Bayesian updating derivation (Advanced)",
                "With θ ~ N(μ_g, σ²_θ) and s | θ ~ N(θ, σ²_ε_g), the "
                "posterior is:\n\n"
                r"$$E[\theta \mid s, g] = \frac{\sigma^2_\varepsilon}{"
                r"\sigma^2_\theta + \sigma^2_\varepsilon} \cdot \mu_g + "
                r"\frac{\sigma^2_\theta}{\sigma^2_\theta + "
                r"\sigma^2_\varepsilon} \cdot s$$"
                "\n\nDefining the shrinkage factor B_g = σ²_ε/(σ²_θ + σ²_ε):\n\n"
                r"$$w(s, g) = B_g \cdot \mu_g + (1 - B_g) \cdot s$$"
                "\n\nThe posterior variance is:\n\n"
                r"$$\text{Var}(\theta \mid s, g) = \frac{\sigma^2_\theta "
                r"\cdot \sigma^2_\varepsilon}{\sigma^2_\theta + "
                r"\sigma^2_\varepsilon}$$"
                "\n\nAs σ²_ε → 0, B_g → 0 and w → s (perfect signal). "
                "As σ²_ε → ∞, B_g → 1 and w → μ_g (pure profiling). "
                "The key comparative static is ∂w/∂s = 1 − B_g: the noisier "
                "the signal, the less individual performance matters."
            ))
        return sections


# ---------------------------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------------------------

def main():
    concept = StatisticalDiscrimination()

    # --- Sidebar ---
    st.sidebar.header("Statistical Discrimination")
    st.sidebar.markdown("---")

    st.sidebar.subheader("True productivity")
    mu_A = st.sidebar.slider("Group A mean (μ_A)", 5.0, 30.0, 15.0, 1.0)
    mu_B = st.sidebar.slider("Group B mean (μ_B)", 5.0, 30.0, 15.0, 1.0)
    sigma_theta = st.sidebar.slider("Productivity SD (σ_θ)", 1.0, 10.0, 4.0, 0.5,
                                    help="Same for both groups")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Signal precision")
    sigma_eps_A = st.sidebar.slider("Group A noise (σ_ε_A)", 0.5, 15.0, 2.0, 0.5,
                                    help="Lower = more precise signal")
    sigma_eps_B = st.sidebar.slider("Group B noise (σ_ε_B)", 0.5, 15.0, 6.0, 0.5,
                                    help="Lower = more precise signal")

    p = {"mu_A": mu_A, "mu_B": mu_B, "sigma_theta": sigma_theta,
         "sigma_eps_A": sigma_eps_A, "sigma_eps_B": sigma_eps_B}
    eq = concept.compute_equilibrium(p)

    # --- Main area ---
    st.title("Statistical Discrimination")
    st.markdown(concept.description)

    st.markdown("---")

    same_means = "identical" if abs(mu_A - mu_B) < 0.1 else "different"
    noisier = ("B" if sigma_eps_B > sigma_eps_A
               else "A" if sigma_eps_A > sigma_eps_B else "neither")
    st.markdown(f"""
**How this app works:** Two groups of workers have {same_means} mean
productivity but employers observe a noisy signal of each worker's
true output. The signal is noisier for Group {noisier}
{"" if noisier == "neither" else f"(σ_ε = {max(sigma_eps_A, sigma_eps_B):.1f} vs {min(sigma_eps_A, sigma_eps_B):.1f})"}.
Employers set wages using Bayesian updating: w = B·μ_group + (1−B)·signal,
where B is the shrinkage factor. The app shows how differential signal noise
leads to different wage schedules — even for workers with the same signal.
""")

    st.markdown("---")

    # --- Headline metrics ---
    st.markdown("**Shrinkage Factors** (weight on group mean vs individual signal)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Group A shrinkage (B_A)", f"{eq['B_A']:.1%}",
              help="Fraction of wage determined by group mean")
    c2.metric("Group B shrinkage (B_B)", f"{eq['B_B']:.1%}",
              help="Fraction of wage determined by group mean")
    c3.metric("Difference (B_B − B_A)", f"{eq['B_B'] - eq['B_A']:+.1%}",
              help="Positive = Group B faces more profiling")

    # --- Plots ---

    # 1. Wage offer functions (full width)
    st.markdown("**Wage Offer as a Function of Signal**")
    fig1, ax1 = plt.subplots(figsize=(9, 5.5))

    # Signal range: cover most of both signal distributions
    s_lo = min(mu_A - 3 * eq["sigma_s_A"], mu_B - 3 * eq["sigma_s_B"])
    s_hi = max(mu_A + 3 * eq["sigma_s_A"], mu_B + 3 * eq["sigma_s_B"])
    s_range = np.linspace(s_lo, s_hi, 500)

    # 45-degree line (perfect information)
    ax1.plot(s_range, s_range, color="#94a3b8", linewidth=1, linestyle=":",
             label="w = s (perfect info)", zorder=1)

    # Group wage functions
    w_A = concept.wage_offer(s_range, "A", eq)
    w_B = concept.wage_offer(s_range, "B", eq)
    ax1.plot(s_range, w_A, color="#2563eb", linewidth=2.5,
             label=f"Group A: w = {eq['B_A']:.0%}·μ_A + {eq['slope_A']:.0%}·s")
    ax1.plot(s_range, w_B, color="#dc2626", linewidth=2.5,
             label=f"Group B: w = {eq['B_B']:.0%}·μ_B + {eq['slope_B']:.0%}·s")

    # Group means as horizontal reference
    ax1.axhline(mu_A, color="#2563eb", linewidth=1, linestyle="--", alpha=0.3)
    ax1.axhline(mu_B, color="#dc2626", linewidth=1, linestyle="--", alpha=0.3)

    # Mark where the two wage functions cross (if they do)
    if eq["s_equal"] is not None and s_lo < eq["s_equal"] < s_hi:
        w_cross = concept.wage_offer(eq["s_equal"], "A", eq)
        ax1.plot(eq["s_equal"], w_cross, "o", color="#7c3aed",
                 markersize=10, zorder=6)
        ax1.text(eq["s_equal"], w_cross + 0.5,
                 f"  Equal at s = {eq['s_equal']:.1f}",
                 color="#7c3aed", fontsize=9, fontweight="bold",
                 va="bottom")

    # Annotate the gap at a high signal
    s_demo = mu_A + 1.5 * sigma_theta
    w_A_demo = concept.wage_offer(s_demo, "A", eq)
    w_B_demo = concept.wage_offer(s_demo, "B", eq)
    if abs(w_A_demo - w_B_demo) > 0.3:
        ax1.annotate("", xy=(s_demo, max(w_A_demo, w_B_demo)),
                     xytext=(s_demo, min(w_A_demo, w_B_demo)),
                     arrowprops=dict(arrowstyle="<->", color="#7c3aed", lw=2))
        mid = (w_A_demo + w_B_demo) / 2
        gap_val = w_A_demo - w_B_demo
        ax1.text(s_demo + 0.3, mid, f"Gap = {gap_val:+.1f}",
                 color="#7c3aed", fontsize=10, fontweight="bold",
                 va="center", ha="left",
                 bbox=dict(facecolor="white", edgecolor="#7c3aed",
                           alpha=0.9, boxstyle="round,pad=0.2"))

    ax1.set_xlabel("Signal (s)", fontsize=11)
    ax1.set_ylabel("Wage offer w(s, g)", fontsize=11)
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(True, alpha=0.2)
    ax1.set_title(
        "Noisier signal → flatter line → more weight on group mean",
        fontsize=11, fontweight="bold")
    fig1.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)

    # 2–3. Wage gap function + wage distributions (side by side)
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**Wage Gap by Signal Level**")
        fig2, ax2 = plt.subplots(figsize=(6, 4.5))

        gap = w_A - w_B
        ax2.plot(s_range, gap, color="#7c3aed", linewidth=2.5)
        ax2.axhline(0, color="black", linewidth=0.5)
        ax2.fill_between(s_range, gap, 0,
                         where=gap > 0, alpha=0.15, color="#2563eb",
                         label="A paid more")
        ax2.fill_between(s_range, gap, 0,
                         where=gap < 0, alpha=0.15, color="#dc2626",
                         label="B paid more")

        if eq["s_equal"] is not None and s_lo < eq["s_equal"] < s_hi:
            ax2.axvline(eq["s_equal"], color="#7c3aed", linewidth=1,
                        linestyle="--", alpha=0.5)
            ax2.text(eq["s_equal"], ax2.get_ylim()[1] * 0.8,
                     f"s = {eq['s_equal']:.1f}",
                     ha="center", fontsize=9, color="#7c3aed")

        ax2.set_xlabel("Signal (s)", fontsize=10)
        ax2.set_ylabel("Wage gap: w_A(s) − w_B(s)", fontsize=10)
        ax2.legend(fontsize=8, loc="best")
        ax2.grid(True, alpha=0.2)

        if abs(mu_A - mu_B) < 0.1 and abs(sigma_eps_A - sigma_eps_B) < 0.1:
            title = "No discrimination — identical groups & signals"
        elif abs(mu_A - mu_B) < 0.1:
            title = "Same means — gap is purely from signal noise"
        else:
            title = "Gap driven by mean differences + signal noise"
        ax2.set_title(title, fontsize=11, fontweight="bold")
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    with col_right:
        st.markdown("**Wage Offer Distributions**")
        fig3, ax3 = plt.subplots(figsize=(6, 4.5))

        # True productivity distributions
        x_range = np.linspace(s_lo, s_hi, 500)
        prod_A = stats.norm.pdf(x_range, mu_A, sigma_theta)
        prod_B = stats.norm.pdf(x_range, mu_B, sigma_theta)

        ax3.fill_between(x_range, prod_A, alpha=0.12, color="#2563eb")
        ax3.fill_between(x_range, prod_B, alpha=0.12, color="#dc2626")
        ax3.plot(x_range, prod_A, color="#2563eb", linewidth=1,
                 linestyle=":", alpha=0.5, label="True θ (A)")
        ax3.plot(x_range, prod_B, color="#dc2626", linewidth=1,
                 linestyle=":", alpha=0.5, label="True θ (B)")

        # Wage distributions
        if eq["wage_sd_A"] > 0.01:
            wage_A = stats.norm.pdf(x_range, eq["wage_mean_A"], eq["wage_sd_A"])
            ax3.plot(x_range, wage_A, color="#2563eb", linewidth=2.5,
                     label=f"Wage A (SD={eq['wage_sd_A']:.1f})")
        if eq["wage_sd_B"] > 0.01:
            wage_B = stats.norm.pdf(x_range, eq["wage_mean_B"], eq["wage_sd_B"])
            ax3.plot(x_range, wage_B, color="#dc2626", linewidth=2.5,
                     label=f"Wage B (SD={eq['wage_sd_B']:.1f})")

        ax3.set_xlabel("Value", fontsize=10)
        ax3.set_ylabel("Density", fontsize=10)
        ax3.legend(fontsize=8, loc="upper right")
        ax3.grid(True, alpha=0.2)
        ax3.set_ylim(bottom=0)
        ax3.set_title("Wages are compressed vs true productivity",
                      fontsize=11, fontweight="bold")
        fig3.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

    st.caption(
        "**Top:** Wage offer as a function of signal for each group. The "
        "45° line represents perfect information (w = s). Each group's "
        "line is rotated toward the group mean — more rotation for the "
        "noisier group. Two workers with the same signal get different wages. "
        "**Bottom left:** The wage gap w_A − w_B at each signal level. "
        "With identical means, high-signal workers from the noisier group "
        "are underpaid and low-signal workers are overpaid. "
        "**Bottom right:** The wage distribution (solid) is compressed "
        "relative to the true productivity distribution (dotted) — more "
        "compression for the noisier group."
    )

    st.markdown("---")

    # --- Metrics ---
    st.markdown("**Model Parameters**")
    c1, c2 = st.columns(2)
    c1.markdown("*Group A*")
    c1.text(f"  Mean productivity   μ_A = {mu_A:.1f}\n"
            f"  Signal noise     σ_ε_A = {sigma_eps_A:.1f}\n"
            f"  Shrinkage          B_A = {eq['B_A']:.3f}\n"
            f"  Wage slope     1 − B_A = {eq['slope_A']:.3f}\n"
            f"  Wage SD              = {eq['wage_sd_A']:.2f}\n"
            f"  Posterior var        = {eq['post_var_A']:.2f}")
    c2.markdown("*Group B*")
    c2.text(f"  Mean productivity   μ_B = {mu_B:.1f}\n"
            f"  Signal noise     σ_ε_B = {sigma_eps_B:.1f}\n"
            f"  Shrinkage          B_B = {eq['B_B']:.3f}\n"
            f"  Wage slope     1 − B_B = {eq['slope_B']:.3f}\n"
            f"  Wage SD              = {eq['wage_sd_B']:.2f}\n"
            f"  Posterior var        = {eq['post_var_B']:.2f}")

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
        page_title="Statistical Discrimination",
        page_icon="📊",
        layout="wide",
    )
    main()
