"""Oaxaca-Blinder Wage Decomposition — Streamlit app.

Visualize how the wage gap between two groups decomposes into an
*explained* component (differences in education) and an *unexplained*
component (differences in returns to education, often interpreted as
potential discrimination).
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from shared.base import SliderSpec, EstimationConcept


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

class OaxacaBlinder(EstimationConcept):
    @property
    def name(self) -> str:
        return "Oaxaca-Blinder Wage Decomposition"

    @property
    def description(self) -> str:
        return (
            "The Oaxaca-Blinder decomposition splits the mean wage gap between "
            "two groups into an *explained* part (due to differences in "
            "characteristics like education) and an *unexplained* part (due to "
            "differences in how those characteristics are rewarded — often "
            "interpreted as a measure of labour-market discrimination)."
        )

    def params(self) -> list[SliderSpec]:
        return [
            SliderSpec("n", "Sample size per group", 50.0, 2000.0, 500.0, 50.0,
                       "Number of workers in each group"),
            SliderSpec("edu_mean_a", "Group A — mean education (yrs)", 8.0, 20.0, 14.0, 0.5,
                       "Average years of schooling for Group A"),
            SliderSpec("edu_mean_b", "Group B — mean education (yrs)", 8.0, 20.0, 13.0, 0.5,
                       "Average years of schooling for Group B"),
            SliderSpec("intercept_a", "Group A — base wage (intercept)", 5.0, 15.0, 8.0, 0.5,
                       "Intercept of the wage equation for Group A ($/hr)"),
            SliderSpec("beta_edu_a", "Group A — return to education", 0.0, 3.0, 1.2, 0.1,
                       "Additional $/hr per year of education for Group A"),
            SliderSpec("intercept_b", "Group B — base wage (intercept)", 5.0, 15.0, 7.0, 0.5,
                       "Intercept of the wage equation for Group B ($/hr)"),
            SliderSpec("beta_edu_b", "Group B — return to education", 0.0, 3.0, 0.9, 0.1,
                       "Additional $/hr per year of education for Group B"),
            SliderSpec("sigma", "Wage noise (σ)", 0.5, 10.0, 3.0, 0.5,
                       "Standard deviation of idiosyncratic wage shocks"),
        ]

    def render(self, params: dict[str, Any], depth: str = "undergraduate") -> None:
        pass  # rendering handled by main()

    def generate_data(self, params: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
        n = int(params["n"])

        edu_a = rng.normal(params["edu_mean_a"], 2.0, size=n)
        wage_a = (params["intercept_a"]
                  + params["beta_edu_a"] * edu_a
                  + rng.normal(0, params["sigma"], size=n))

        edu_b = rng.normal(params["edu_mean_b"], 2.0, size=n)
        wage_b = (params["intercept_b"]
                  + params["beta_edu_b"] * edu_b
                  + rng.normal(0, params["sigma"], size=n))

        return {
            "edu_a": edu_a, "wage_a": wage_a,
            "edu_b": edu_b, "wage_b": wage_b,
        }

    def estimate(self, data: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        """Run OLS on each group and perform the Oaxaca-Blinder decomposition."""
        n = len(data["edu_a"])
        ones = np.ones(n)

        # Design matrices [1, edu]
        X_a = np.column_stack([ones, data["edu_a"]])
        X_b = np.column_stack([ones, data["edu_b"]])

        # OLS
        beta_a, _, _, _ = np.linalg.lstsq(X_a, data["wage_a"], rcond=None)
        beta_b, _, _, _ = np.linalg.lstsq(X_b, data["wage_b"], rcond=None)

        # Mean characteristics
        Xbar_a = X_a.mean(axis=0)  # [1, mean_edu_a]
        Xbar_b = X_b.mean(axis=0)

        # Mean wages
        ybar_a = data["wage_a"].mean()
        ybar_b = data["wage_b"].mean()
        total_gap = ybar_a - ybar_b

        # Decomposition using Group A coefficients as reference
        explained = (Xbar_a - Xbar_b) @ beta_a
        unexplained = Xbar_b @ (beta_a - beta_b)

        # Per-variable detail
        var_names = ["Intercept", "Education"]
        explained_detail = (Xbar_a - Xbar_b) * beta_a
        unexplained_detail = Xbar_b * (beta_a - beta_b)

        return {
            "beta_a": beta_a,
            "beta_b": beta_b,
            "Xbar_a": Xbar_a,
            "Xbar_b": Xbar_b,
            "ybar_a": ybar_a,
            "ybar_b": ybar_b,
            "total_gap": total_gap,
            "explained": explained,
            "unexplained": unexplained,
            "var_names": var_names,
            "explained_detail": explained_detail,
            "unexplained_detail": unexplained_detail,
        }

    def educational_sections(self, depth: str = "undergraduate") -> list:
        sections = [
            ("What is the Oaxaca-Blinder decomposition?",
             "The **Oaxaca-Blinder decomposition** (1973) is a statistical method "
             "for decomposing the difference in mean outcomes (typically wages) "
             "between two groups.\n\n"
             "Suppose each group has a wage equation:\n\n"
             r"$$w_A = X_A \beta_A + \varepsilon_A \qquad w_B = X_B \beta_B + \varepsilon_B$$"
             "\n\nThe mean wage gap is:\n\n"
             r"$$\Delta\bar{w} = \bar{w}_A - \bar{w}_B = "
             r"\underbrace{(\bar{X}_A - \bar{X}_B)'\hat{\beta}_A}_{\text{Explained}} + "
             r"\underbrace{\bar{X}_B'(\hat{\beta}_A - \hat{\beta}_B)}_{\text{Unexplained}}$$"
             "\n\n- **Explained**: gap due to group differences in characteristics "
             "(education, etc.)\n"
             "- **Unexplained**: gap due to different *returns* to the same "
             "characteristics — often interpreted as a discrimination measure"),
            ("What does 'unexplained' really mean?",
             "The unexplained component is sometimes called the 'discrimination "
             "coefficient', but this interpretation requires caution:\n\n"
             "- It captures differences in **returns** to characteristics, not just "
             "discrimination\n"
             "- **Omitted variables** (e.g., experience, occupation choice, hours "
             "flexibility) can inflate or deflate the unexplained part\n"
             "- The decomposition is **not causal** — it's a descriptive accounting "
             "exercise\n"
             "- The choice of **reference group** matters (using Group A vs Group B "
             "coefficients as the benchmark gives different results)"),
            ("Why does the reference group matter?",
             "The decomposition uses one group's coefficients as the 'non-discriminatory' "
             "benchmark. Using Group A:\n\n"
             r"$$\text{Explained} = (\bar{X}_A - \bar{X}_B)'\hat{\beta}_A$$"
             "\n\nUsing Group B:\n\n"
             r"$$\text{Explained} = (\bar{X}_A - \bar{X}_B)'\hat{\beta}_B$$"
             "\n\nThese give different results because the counterfactual is different. "
             "This app uses Group A as the reference (the more common convention when "
             "Group A is the higher-wage group)."),
        ]
        if depth == "graduate":
            sections.append((
                "Cotton-Neumark pooled decomposition",
                "To avoid the index number problem of choosing a reference group, "
                "**Cotton (1988)** and **Neumark (1988)** proposed using the pooled "
                "regression coefficients β* as the non-discriminatory benchmark:\n\n"
                r"$$\Delta\bar{w} = (\bar{X}_A - \bar{X}_B)'\hat{\beta}^* + "
                r"\bar{X}_A'(\hat{\beta}_A - \hat{\beta}^*) + "
                r"\bar{X}_B'(\hat{\beta}^* - \hat{\beta}_B)$$"
                "\n\nThis three-fold decomposition separates the unexplained "
                "component into an *advantage* to Group A and a *disadvantage* "
                "to Group B, relative to the pooled structure."
            ))
        return sections


# ---------------------------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------------------------

def main():
    concept = OaxacaBlinder()

    # --- Sidebar ---
    st.sidebar.header("Oaxaca-Blinder Decomposition")
    st.sidebar.markdown("---")

    st.sidebar.subheader("Sample")
    n = st.sidebar.slider("Sample size per group", 50, 2000, 500, 50)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Group A (higher-wage)")
    edu_mean_a = st.sidebar.slider("Mean education (yrs)", 8.0, 20.0, 14.0, 0.5, key="edu_a")
    intercept_a = st.sidebar.slider("Base wage (intercept)", 5.0, 15.0, 8.0, 0.5, key="int_a")
    beta_edu_a = st.sidebar.slider("Return to education ($/yr)", 0.0, 3.0, 1.2, 0.1, key="bedu_a")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Group B (lower-wage)")
    edu_mean_b = st.sidebar.slider("Mean education (yrs)", 8.0, 20.0, 13.0, 0.5, key="edu_b")
    intercept_b = st.sidebar.slider("Base wage (intercept)", 5.0, 15.0, 7.0, 0.5, key="int_b")
    beta_edu_b = st.sidebar.slider("Return to education ($/yr)", 0.0, 3.0, 0.9, 0.1, key="bedu_b")

    st.sidebar.markdown("---")
    sigma = st.sidebar.slider("Wage noise (σ)", 0.5, 10.0, 3.0, 0.5)
    depth = st.sidebar.radio("Depth", ["undergraduate", "graduate"],
                             format_func=lambda x: x.title())
    use_seed = st.sidebar.checkbox("Fix random seed", value=True)
    seed = 42 if use_seed else None

    p = {
        "n": n,
        "edu_mean_a": edu_mean_a, "edu_mean_b": edu_mean_b,
        "intercept_a": intercept_a, "beta_edu_a": beta_edu_a,
        "intercept_b": intercept_b, "beta_edu_b": beta_edu_b,
        "sigma": sigma,
    }

    rng = np.random.default_rng(seed)
    data = concept.generate_data(p, rng)
    result = concept.estimate(data, p)

    # --- Main area ---
    st.title("Oaxaca-Blinder Wage Decomposition")
    st.markdown(
        concept.description + " The decomposition formula is:"
    )
    st.latex(
        r"\Delta\bar{w} = \underbrace{(\bar{X}_A - \bar{X}_B)'\hat{\beta}_A}"
        r"_{\text{Explained}} + "
        r"\underbrace{\bar{X}_B'(\hat{\beta}_A - \hat{\beta}_B)}"
        r"_{\text{Unexplained}}"
    )

    st.markdown("---")

    st.markdown("""
**How this app works:** The sidebar sliders define the *population* wage
equations for two groups — each group has its own intercept and return to
education. A random sample is drawn from these populations, OLS is run
separately on each group, and the Oaxaca-Blinder decomposition is applied
to the *estimated* coefficients. Because the estimates are based on a finite
sample, they won't match the true parameters exactly — try increasing the
sample size or decreasing the noise to see them converge.
""")

    st.markdown("---")

    # Headline metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Mean wage — Group A", f"${result['ybar_a']:.2f}/hr")
    c2.metric("Mean wage — Group B", f"${result['ybar_b']:.2f}/hr")
    c3.metric("Raw gap (A − B)", f"${result['total_gap']:.2f}/hr")

    c1, c2 = st.columns(2)
    expl_pct = (result["explained"] / result["total_gap"] * 100
                if abs(result["total_gap"]) > 0.001 else 0)
    unexpl_pct = (result["unexplained"] / result["total_gap"] * 100
                  if abs(result["total_gap"]) > 0.001 else 0)
    c1.metric("Explained", f"${result['explained']:.2f}/hr ({expl_pct:.0f}%)")
    c2.metric("Unexplained", f"${result['unexplained']:.2f}/hr ({unexpl_pct:.0f}%)")

    st.markdown("---")

    # --- Plots ---
    beta_a = result["beta_a"]  # [intercept, slope]
    beta_b = result["beta_b"]
    Xbar_a = result["Xbar_a"]  # [1, mean_edu_a]
    Xbar_b = result["Xbar_b"]

    col_left, col_right = st.columns(2)

    # 1. Decomposition bar chart
    with col_left:
        st.markdown("**Wage Gap Decomposition**")
        fig1, ax1 = plt.subplots(figsize=(6, 4.5))

        total = result["total_gap"]
        expl = result["explained"]
        unexpl = result["unexplained"]

        bars = ax1.barh(
            ["Unexplained\n(coefficients)", "Explained\n(characteristics)", "Total gap"],
            [unexpl, expl, total],
            color=["#ef4444", "#3b82f6", "#64748b"],
            edgecolor="white",
            height=0.5,
        )

        for bar, val in zip(bars, [unexpl, expl, total]):
            x_pos = bar.get_width()
            ax1.text(x_pos + 0.1 if x_pos >= 0 else x_pos - 0.1,
                     bar.get_y() + bar.get_height() / 2,
                     f"${val:.2f}", va="center",
                     ha="left" if x_pos >= 0 else "right",
                     fontsize=10, fontweight="bold")

        ax1.set_xlabel("$/hr", fontsize=10)
        ax1.axvline(0, color="black", linewidth=0.5)
        ax1.grid(True, alpha=0.2, axis="x")
        ax1.set_title("Decomposition of mean wage gap", fontsize=12, fontweight="bold")
        fig1.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

    # 2. Scatter plot with decomposition annotations
    with col_right:
        st.markdown("**Wage vs Education**")
        fig2, ax2 = plt.subplots(figsize=(6, 4.5))
        ax2.scatter(data["edu_a"], data["wage_a"], alpha=0.15, s=8,
                    color="#93c5fd", rasterized=True)
        ax2.scatter(data["edu_b"], data["wage_b"], alpha=0.15, s=8,
                    color="#fca5a5", rasterized=True)

        edu_range = np.linspace(
            min(data["edu_a"].min(), data["edu_b"].min()) - 0.5,
            max(data["edu_a"].max(), data["edu_b"].max()) + 0.5,
            200,
        )

        # Estimated regression lines
        line_a = beta_a[0] + beta_a[1] * edu_range
        line_b = beta_b[0] + beta_b[1] * edu_range

        ax2.plot(edu_range, line_b, color="#dc2626", linewidth=2, label="Group B (est.)")
        ax2.plot(edu_range, line_a, color="#2563eb", linewidth=2, label="Group A (est.)")

        # Three key points:
        #   P_B  = Group B line at X̄_B_edu (red dot)
        #   P_CF = Group A line at X̄_B_edu (purple square — counterfactual)
        #   P_A  = Group A line at X̄_A_edu (blue dot)
        x_b = Xbar_b[1]
        x_a = Xbar_a[1]
        y_b = beta_b[0] + beta_b[1] * x_b
        y_cf = beta_a[0] + beta_a[1] * x_b
        y_a = beta_a[0] + beta_a[1] * x_a

        ax2.plot(x_b, y_b, "o", color="#dc2626", markersize=9, zorder=6)
        ax2.plot(x_b, y_cf, "s", color="#7c3aed", markersize=9, zorder=6)
        ax2.plot(x_a, y_a, "o", color="#2563eb", markersize=9, zorder=6)

        # Unexplained: vertical arrow at X̄_B from B's line to A's line
        if abs(y_cf - y_b) > 0.05:
            ax2.annotate(
                "", xy=(x_b, y_cf), xytext=(x_b, y_b),
                arrowprops=dict(arrowstyle="<->", color="#7c3aed", lw=2),
                zorder=5,
            )
            ax2.text(x_b - 0.3, (y_b + y_cf) / 2, "Unexplained",
                     color="#7c3aed", fontsize=8, fontweight="bold",
                     ha="right", va="center",
                     bbox=dict(facecolor="white", edgecolor="#7c3aed",
                               alpha=0.9, boxstyle="round,pad=0.2"))

        # Explained: arrow along Group A's line from X̄_B to X̄_A
        if abs(x_a - x_b) > 0.05 or abs(y_a - y_cf) > 0.05:
            ax2.annotate(
                "", xy=(x_a, y_a), xytext=(x_b, y_cf),
                arrowprops=dict(arrowstyle="<->", color="#2563eb", lw=2),
                zorder=5,
            )
            mid_x = (x_b + x_a) / 2
            mid_y = (y_cf + y_a) / 2
            ax2.text(mid_x + 0.3, mid_y, "Explained",
                     color="#2563eb", fontsize=8, fontweight="bold",
                     ha="left", va="center",
                     bbox=dict(facecolor="white", edgecolor="#2563eb",
                               alpha=0.9, boxstyle="round,pad=0.2"))

        ax2.set_xlabel("Education (years)", fontsize=10)
        ax2.set_ylabel("Wage ($/hr)", fontsize=10)
        ax2.legend(fontsize=8, loc="upper left")
        ax2.grid(True, alpha=0.2)
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    st.caption(
        "The purple square is the **counterfactual**: Group B's mean education "
        "evaluated on Group A's regression line. The vertical arrow "
        "(**unexplained**) shows the gap between the two lines at Group B's "
        "mean education — due to different returns. The diagonal arrow "
        "(**explained**) shows the movement along Group A's line from Group B's "
        "mean education to Group A's — due to different characteristics."
    )

    # --- OLS coefficient comparison ---
    st.markdown("---")
    st.markdown("**Estimated Coefficients (OLS)**")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("*Group A*")
        for name, true_val, est_val in zip(
            ["Intercept", "Education"],
            [p["intercept_a"], p["beta_edu_a"]],
            result["beta_a"],
        ):
            st.text(f"  {name:12s}  true = {true_val:6.2f}   est = {est_val:6.2f}")

    with col2:
        st.markdown("*Group B*")
        for name, true_val, est_val in zip(
            ["Intercept", "Education"],
            [p["intercept_b"], p["beta_edu_b"]],
            result["beta_b"],
        ):
            st.text(f"  {name:12s}  true = {true_val:6.2f}   est = {est_val:6.2f}")

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
        page_title="Oaxaca-Blinder Decomposition",
        page_icon="📊",
        layout="wide",
    )
    main()
