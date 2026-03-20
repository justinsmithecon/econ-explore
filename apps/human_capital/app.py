"""Human Capital & Returns to Schooling — Streamlit app.

Compare lifetime income streams with and without additional education,
accounting for direct costs (tuition) and opportunity costs (foregone
earnings). Compute the net present value and internal rate of return
of the education investment.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import streamlit as st

from shared.base import SliderSpec, InteractiveConcept


class HumanCapital(InteractiveConcept):
    """Human capital investment decision.

    Compares two income paths:
      Path A (no extra schooling): earn w_0 immediately, with growth g
      Path B (extra schooling): pay tuition and forego earnings for k years,
             then earn w_1 (> w_0) with the same growth rate g

    All values are computed in real (inflation-adjusted) terms.
    """

    @property
    def name(self) -> str:
        return "Human Capital & Returns to Schooling"

    @property
    def description(self) -> str:
        return (
            "Education is an **investment in human capital**: you pay direct "
            "costs (tuition) and forego earnings today in exchange for higher "
            "wages in the future. Whether the investment pays off depends on "
            "how much wages increase, how long you work afterward, and how "
            "you value future vs present income (the discount rate)."
        )

    def params(self) -> list[SliderSpec]:
        return [
            SliderSpec("w_0", "Annual wage without extra schooling", 20000, 100000, 35000, 1000,
                       "What you'd earn if you started working now"),
            SliderSpec("w_1", "Annual wage with extra schooling", 20000, 150000, 50000, 1000,
                       "What you'd earn after completing additional education"),
            SliderSpec("k", "Years of extra schooling", 1, 8, 4, 1,
                       "How many additional years of education"),
            SliderSpec("tuition", "Annual tuition", 0, 60000, 15000, 1000,
                       "Direct cost per year of schooling"),
            SliderSpec("g", "Annual wage growth (%)", 0.0, 5.0, 2.0, 0.5,
                       "Real annual wage growth rate (both paths)"),
            SliderSpec("r", "Discount rate (%)", 0.0, 15.0, 5.0, 0.5,
                       "Rate at which you discount future income"),
            SliderSpec("T", "Career horizon (years)", 20, 50, 40, 1,
                       "Total years from now until retirement"),
        ]

    def render(self, params: dict[str, Any], depth: str = "undergraduate") -> None:
        pass

    def compute(self, params: dict[str, Any]) -> dict[str, Any]:
        w_0 = params["w_0"]
        w_1 = params["w_1"]
        k = int(params["k"])
        tuition = params["tuition"]
        g = params["g"] / 100
        r = params["r"] / 100
        T = int(params["T"])

        years = np.arange(T)

        # Path A: work from year 0
        income_a = np.array([w_0 * (1 + g) ** t for t in years])

        # Path B: school for k years (pay tuition, earn nothing), then work
        income_b = np.zeros(T)
        for t in years:
            if t < k:
                income_b[t] = -tuition  # negative = cost
            else:
                income_b[t] = w_1 * (1 + g) ** (t - k)

        # Discount factors
        discount = np.array([(1 + r) ** (-t) for t in years])

        # Present values
        pv_a = (income_a * discount).sum()
        pv_b_earnings = np.where(income_b > 0, income_b, 0)
        pv_b = (income_b * discount).sum()

        npv_education = pv_b - pv_a

        # Cumulative discounted earnings
        cum_a = np.cumsum(income_a * discount)
        cum_b = np.cumsum(income_b * discount)

        # Break-even year (discounted — consistent with NPV)
        diff = cum_b - cum_a
        crossover_indices = np.where(diff[:-1] * diff[1:] < 0)[0]
        if len(crossover_indices) > 0:
            # Linear interpolation for the crossover
            i = crossover_indices[0]
            frac = -diff[i] / (diff[i + 1] - diff[i])
            breakeven = i + frac
        elif diff[-1] >= 0 and diff[0] < 0:
            breakeven = float(T)  # hasn't crossed yet within horizon
        else:
            breakeven = None

        # Total costs
        total_tuition = tuition * k
        total_opportunity = sum(w_0 * (1 + g) ** t for t in range(k))
        total_cost = total_tuition + total_opportunity

        # Internal rate of return: find r such that NPV(education) = 0
        # Net cash flow of choosing B over A
        net_cf = income_b - income_a
        irr = _compute_irr(net_cf)

        # Discounted annual income
        pv_income_a = income_a * discount
        pv_income_b = income_b * discount

        return {
            "years": years,
            "income_a": income_a,
            "income_b": income_b,
            "pv_income_a": pv_income_a,
            "pv_income_b": pv_income_b,
            "cum_a": cum_a,
            "cum_b": cum_b,
            "pv_a": pv_a,
            "pv_b": pv_b,
            "npv_education": npv_education,
            "breakeven": breakeven,
            "total_tuition": total_tuition,
            "total_opportunity": total_opportunity,
            "total_cost": total_cost,
            "irr": irr,
            "k": k,
        }

    def educational_sections(self, depth: str = "undergraduate") -> list:
        sections = [
            ("How to think about education as an investment",
             "Going to school is like buying an asset that pays off over your "
             "working life. The costs are:\n\n"
             "- **Direct costs**: tuition, books, fees\n"
             "- **Opportunity costs**: the wages you *give up* while studying\n\n"
             "The benefit is **higher lifetime earnings**. The investment is "
             "worthwhile if the present value of the extra earnings exceeds "
             "the present value of the costs."),
            ("What is the discount rate?",
             "The **discount rate** reflects how much you prefer money today "
             "over money in the future. A higher discount rate means future "
             "earnings are worth less to you today.\n\n"
             "- At r = 0%: a dollar 20 years from now is worth a dollar today\n"
             "- At r = 5%: a dollar 20 years from now is worth only $0.38 today\n"
             "- At r = 10%: a dollar 20 years from now is worth only $0.15 today\n\n"
             "Higher discount rates make education look *less* attractive because "
             "the costs are paid now but the benefits come later."),
            ("What is the internal rate of return (IRR)?",
             "The **IRR** is the discount rate at which the education investment "
             "exactly breaks even (NPV = 0). If the IRR exceeds your personal "
             "discount rate (or the market interest rate), the investment is "
             "worthwhile.\n\n"
             "Typical estimates of the IRR to a college degree in the US are "
             "around 10–15%, well above most discount rates — suggesting "
             "college is a good financial investment on average."),
        ]
        if depth == "graduate":
            sections.append((
                "Present value formula",
                "The net present value of the education investment is:\n\n"
                r"$$\text{NPV} = \sum_{t=0}^{k-1} \frac{-C_t - w_0(1+g)^t}{(1+r)^t} "
                r"+ \sum_{t=k}^{T-1} \frac{w_1(1+g)^{t-k} - w_0(1+g)^t}{(1+r)^t}$$"
                "\n\nwhere C_t is the direct cost (tuition) in year t, w_0 is the "
                "no-schooling wage, w_1 is the post-schooling wage, k is years "
                "of schooling, g is the wage growth rate, and r is the discount rate.\n\n"
                "The first sum captures the **investment period** (tuition + foregone "
                "earnings). The second sum captures the **payoff period** (wage premium "
                "net of what you would have earned anyway)."
            ))
        return sections


def _compute_irr(cash_flows: np.ndarray) -> float | None:
    """Compute IRR by bisection on NPV(r) = 0."""
    def npv_at(r):
        return sum(cf / (1 + r) ** t for t, cf in enumerate(cash_flows))

    lo, hi = -0.5, 2.0
    npv_lo, npv_hi = npv_at(lo), npv_at(hi)
    if npv_lo * npv_hi > 0:
        return None

    for _ in range(200):
        mid = (lo + hi) / 2
        npv_mid = npv_at(mid)
        if abs(npv_mid) < 0.01:
            return mid
        if npv_lo * npv_mid < 0:
            hi = mid
            npv_hi = npv_mid
        else:
            lo = mid
            npv_lo = npv_mid
    return (lo + hi) / 2


# ---------------------------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------------------------

def main():
    concept = HumanCapital()

    # --- Sidebar ---
    st.sidebar.header("Returns to Schooling")
    st.sidebar.markdown("---")

    st.sidebar.subheader("Earnings")
    w_0 = st.sidebar.slider("Wage without extra schooling ($/yr)",
                            20000, 100000, 35000, 1000)
    w_1 = st.sidebar.slider("Wage with extra schooling ($/yr)",
                            20000, 150000, 50000, 1000)
    g = st.sidebar.slider("Annual wage growth (%)", 0.0, 5.0, 2.0, 0.5)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Schooling costs")
    k = st.sidebar.slider("Years of extra schooling", 1, 8, 4, 1)
    tuition = st.sidebar.slider("Annual tuition ($/yr)", 0, 60000, 15000, 1000)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Discounting")
    r = st.sidebar.slider("Discount rate (%)", 0.0, 15.0, 5.0, 0.5)
    T = st.sidebar.slider("Career horizon (years)", 20, 50, 40, 1)

    st.sidebar.markdown("---")
    depth = st.sidebar.radio("Depth", ["undergraduate", "graduate"],
                             format_func=lambda x: x.title())

    p = {"w_0": w_0, "w_1": w_1, "k": k, "tuition": tuition,
         "g": g, "r": r, "T": T}
    result = concept.compute(p)

    # --- Main area ---
    st.title("Human Capital & Returns to Schooling")
    st.markdown(concept.description)

    st.markdown("---")

    st.markdown("""
**How this app works:** The sidebar defines two income paths — one where you
start working immediately, and one where you spend additional years in school
(paying tuition and forgoing earnings) then earn a higher wage. The app
compares the two paths over the chosen career horizon, computing the present
value of each, the break-even point, and the internal rate of return on the
education investment. All values are discounted at the chosen discount rate.
""")

    st.markdown("---")

    # --- Plots ---
    years = result["years"]
    kk = result["k"]

    # 1–2. Nominal and PV income side by side
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**Annual Income (Nominal)**")
        fig1, ax1 = plt.subplots(figsize=(6, 4.5))

        ax1.fill_between(years, result["income_a"], alpha=0.15, color="#2563eb")
        ax1.plot(years, result["income_a"], color="#2563eb", linewidth=2,
                 label="Work now")
        ax1.fill_between(years, np.maximum(result["income_b"], 0),
                         alpha=0.15, color="#dc2626")
        ax1.plot(years, np.maximum(result["income_b"], 0), color="#dc2626",
                 linewidth=2, label="Extra schooling")

        school_years = years[:kk]
        ax1.fill_between(school_years, 0, result["income_a"][:kk],
                         color="#f59e0b", alpha=0.2, label="Opportunity cost")
        if tuition > 0:
            ax1.bar(school_years, [-tuition] * kk,
                    color="#ef4444", alpha=0.4, width=0.8, label="Tuition")

        ax1.axvline(kk, color="#64748b", linestyle=":", linewidth=1, alpha=0.5)

        ax1.set_xlabel("Years from now", fontsize=10)
        ax1.set_ylabel("$/year", fontsize=10)
        ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax1.legend(fontsize=8, loc="lower right")
        ax1.grid(True, alpha=0.2)
        ax1.set_title("Nominal income", fontsize=12, fontweight="bold")
        fig1.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

    with col_right:
        st.markdown("**Annual Income (Present Value)**")
        fig2, ax2 = plt.subplots(figsize=(6, 4.5))

        pv_a = result["pv_income_a"]
        pv_b = result["pv_income_b"]

        ax2.fill_between(years, pv_a, alpha=0.15, color="#2563eb")
        ax2.plot(years, pv_a, color="#2563eb", linewidth=2, label="Work now")
        ax2.fill_between(years, np.maximum(pv_b, 0), alpha=0.15, color="#dc2626")
        ax2.plot(years, np.maximum(pv_b, 0), color="#dc2626", linewidth=2,
                 label="Extra schooling")

        school_years = years[:kk]
        ax2.fill_between(school_years, 0, pv_a[:kk],
                         color="#f59e0b", alpha=0.2, label="Opp. cost (PV)")
        if tuition > 0:
            ax2.bar(school_years, pv_b[:kk],
                    color="#ef4444", alpha=0.4, width=0.8, label="Tuition (PV)")

        ax2.axvline(kk, color="#64748b", linestyle=":", linewidth=1, alpha=0.5)

        ax2.set_xlabel("Years from now", fontsize=10)
        ax2.set_ylabel("PV $/year", fontsize=10)
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax2.legend(fontsize=8, loc="center right")
        ax2.grid(True, alpha=0.2)
        ax2.set_title("Discounted income", fontsize=12, fontweight="bold")
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    # 3. Cumulative PV (full width)
    st.markdown("**Cumulative Present Value**")
    fig3, ax3 = plt.subplots(figsize=(9, 4.5))

    ax3.plot(years, result["cum_a"] / 1000, color="#2563eb", linewidth=2,
             label="Work now")
    ax3.plot(years, result["cum_b"] / 1000, color="#dc2626", linewidth=2,
             label="Extra schooling")

    if result["breakeven"] is not None and result["breakeven"] < T:
        be = result["breakeven"]
        be_val = np.interp(be, years, result["cum_a"] / 1000)
        ax3.plot(be, be_val, "o", color="#7c3aed", markersize=10, zorder=5)
        ax3.annotate(f"Break-even: {be:.1f} years",
                     xy=(be, be_val), xytext=(be + 3, be_val * 0.75),
                     fontsize=10, color="#7c3aed", fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color="#7c3aed", lw=1.5))

    ax3.set_xlabel("Years from now", fontsize=11)
    ax3.set_ylabel("Cumulative PV ($K)", fontsize=11)
    ax3.legend(fontsize=9, loc="upper left")
    ax3.grid(True, alpha=0.2)
    ax3.set_title("Cumulative present value comparison", fontsize=13, fontweight="bold")
    fig3.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

    st.caption(
        "**Top left:** Nominal annual income — the raw earning streams. "
        "**Top right:** Present value of annual income — future dollars discounted "
        "to today. Note how the schooling premium shrinks in PV terms. "
        "**Bottom:** Cumulative PV. The purple dot marks the discounted break-even "
        "point, consistent with the NPV calculation."
    )

    st.markdown("---")

    # --- Metrics ---
    st.markdown("**Investment Analysis**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total tuition", f"${result['total_tuition']:,.0f}")
    c2.metric("Foregone earnings", f"${result['total_opportunity']:,.0f}")
    c3.metric("Total investment", f"${result['total_cost']:,.0f}")
    if result["breakeven"] is not None and result["breakeven"] < T:
        c4.metric("Break-even", f"{result['breakeven']:.1f} yrs")
    else:
        c4.metric("Break-even", "—")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PV (work now)", f"${result['pv_a']:,.0f}")
    c2.metric("PV (extra schooling)", f"${result['pv_b']:,.0f}")
    if result["npv_education"] >= 0:
        c3.metric("NPV of education", f"${result['npv_education']:,.0f}")
    else:
        c3.metric("NPV of education", f"-${abs(result['npv_education']):,.0f}")
    if result["irr"] is not None:
        c4.metric("IRR", f"{result['irr']:.1%}")
    else:
        c4.metric("IRR", "N/A")

    if result["npv_education"] >= 0:
        st.success(
            f"Education investment has **positive NPV** at a {r}% discount rate. "
            f"The IRR of {result['irr']:.1%} exceeds the discount rate."
            if result["irr"] is not None and result["irr"] > r / 100
            else f"Education investment has positive NPV."
        )
    else:
        st.warning(
            f"Education investment has **negative NPV** at a {r}% discount rate. "
            f"The costs outweigh the discounted benefits."
        )


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
        page_title="Human Capital & Returns to Schooling",
        page_icon="📊",
        layout="wide",
    )
    main()
