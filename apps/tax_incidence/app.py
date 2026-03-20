"""Tax Incidence — Streamlit app.

Visualize how a tax creates a wedge between the price buyers pay
and the price sellers receive, and how the economic burden splits between
the two sides depending on the relative elasticities of supply and demand.
Supports both per-unit and ad valorem taxes.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import streamlit as st

from shared.base import SliderSpec, SelectSpec, EquilibriumConcept


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

class TaxIncidence(EquilibriumConcept):
    """Tax incidence with linear supply and demand.

    Demand: P = a_d - b_d * Q   (downward sloping)
    Supply: P = a_s + b_s * Q   (upward sloping)

    Per-unit tax t:    P_buyer = P_seller + t
    Ad valorem tax τ:  P_buyer = P_seller * (1 + τ)
    """

    @property
    def name(self) -> str:
        return "Tax Incidence"

    @property
    def description(self) -> str:
        return (
            "Tax incidence describes how the economic burden of a tax is split "
            "between buyers and sellers. A key insight is that the **statutory "
            "incidence** (who legally pays the tax) does not determine the "
            "**economic incidence** (who actually bears the burden) — that "
            "depends on the relative elasticities of supply and demand."
        )

    def params(self) -> list:
        return [
            SliderSpec("a_d", "Demand intercept (a)", 10.0, 50.0, 30.0, 1.0,
                       "Price-axis intercept of the demand curve"),
            SliderSpec("b_d", "Demand slope (b)", 0.1, 3.0, 1.0, 0.1,
                       "Steepness of demand curve (higher = more inelastic)"),
            SliderSpec("a_s", "Supply intercept (c)", 0.0, 20.0, 2.0, 1.0,
                       "Price-axis intercept of the supply curve"),
            SliderSpec("b_s", "Supply slope (d)", 0.1, 3.0, 1.0, 0.1,
                       "Steepness of supply curve (higher = more inelastic)"),
            SelectSpec("tax_type", "Tax type",
                       ["Per-unit", "Ad valorem"],
                       help_text="Per-unit: fixed $ per unit; Ad valorem: percentage of price"),
            SliderSpec("tax", "Per-unit tax (t)", 0.0, 20.0, 5.0, 0.5,
                       "Tax per unit of the good"),
            SliderSpec("tax_rate", "Ad valorem rate (τ)", 0.0, 1.0, 0.20, 0.01,
                       "Tax as a fraction of price (e.g. 0.20 = 20%)"),
            SelectSpec("statutory", "Tax levied on",
                       ["Sellers", "Buyers"],
                       help_text="Who legally remits the tax (doesn't affect economic incidence)"),
        ]

    def render(self, params: dict[str, Any], depth: str = "undergraduate") -> None:
        pass

    def compute_curves(self, params: dict[str, Any]) -> dict[str, Any]:
        a_d, b_d = params["a_d"], params["b_d"]
        a_s, b_s = params["a_s"], params["b_s"]

        q_max = a_d / b_d  # where demand hits zero
        q = np.linspace(0, q_max, 300)

        demand = a_d - b_d * q
        supply = a_s + b_s * q

        return {"q": q, "demand": demand, "supply": supply}

    def compute_equilibrium(self, params: dict[str, Any]) -> dict[str, Any]:
        a_d, b_d = params["a_d"], params["b_d"]
        a_s, b_s = params["a_s"], params["b_s"]
        tax_type = params.get("tax_type", "Per-unit")

        # Pre-tax equilibrium: a_d - b_d*Q = a_s + b_s*Q
        Q_star = (a_d - a_s) / (b_d + b_s)
        P_star = a_d - b_d * Q_star

        if tax_type == "Ad valorem":
            tau = params.get("tax_rate", 0)

            # P_buyer = P_seller * (1 + τ)
            # P_buyer on demand curve: a_d - b_d*Q
            # P_seller on supply curve: a_s + b_s*Q
            # => a_d - b_d*Q = (a_s + b_s*Q) * (1 + τ)
            denom = b_d + b_s * (1 + tau)
            if denom > 0:
                Q_tax = (a_d - a_s * (1 + tau)) / denom
            else:
                Q_tax = 0
            Q_tax = max(Q_tax, 0.0)

            P_buyer = a_d - b_d * Q_tax
            P_seller = a_s + b_s * Q_tax
            t_effective = P_buyer - P_seller  # = τ * P_seller

            tax_revenue = tau * P_seller * Q_tax if Q_tax > 0 else 0
        else:
            t = params.get("tax", 0)

            # P_buyer = P_seller + t
            # => a_d - b_d*Q = a_s + b_s*Q + t
            Q_tax = (a_d - a_s - t) / (b_d + b_s)
            Q_tax = max(Q_tax, 0.0)

            P_buyer = a_d - b_d * Q_tax
            P_seller = a_s + b_s * Q_tax
            t_effective = t

            tax_revenue = t * Q_tax if Q_tax > 0 else 0

        # Burden
        buyer_burden = P_buyer - P_star
        seller_burden = P_star - P_seller

        total_burden = buyer_burden + seller_burden
        buyer_share = buyer_burden / total_burden if total_burden > 0 else 0
        seller_share = seller_burden / total_burden if total_burden > 0 else 0

        # DWL = area of triangle between S and D from Q_tax to Q_star
        # = 0.5 * (Q_star - Q_tax) * (P_buyer - P_seller)
        dwl = 0.5 * (Q_star - Q_tax) * (P_buyer - P_seller) if Q_tax > 0 else 0

        # Elasticities at pre-tax equilibrium
        if Q_star > 0:
            elasticity_d = abs((1 / b_d) * P_star / Q_star)
            elasticity_s = (1 / b_s) * P_star / Q_star
        else:
            elasticity_d = 0
            elasticity_s = 0

        return {
            "Q_star": Q_star, "P_star": P_star,
            "Q_tax": Q_tax,
            "P_buyer": P_buyer, "P_seller": P_seller,
            "t_effective": t_effective,
            "buyer_burden": buyer_burden, "seller_burden": seller_burden,
            "buyer_share": buyer_share, "seller_share": seller_share,
            "tax_revenue": tax_revenue, "dwl": dwl,
            "elasticity_d": elasticity_d, "elasticity_s": elasticity_s,
        }

    def educational_sections(self) -> list:
        sections = [
            ("Why doesn't statutory incidence matter?",
             "Imagine a \\$5 tax on sellers. The supply curve shifts up by \\$5 — "
             "sellers need \\$5 more per unit to cover the tax. But the new "
             "equilibrium price doesn't rise by the full \\$5; it rises by less "
             "because buyers reduce their quantity demanded.\n\n"
             "Now imagine the same \\$5 tax on buyers instead. The demand curve "
             "shifts down by \\$5 — buyers are willing to pay \\$5 less since "
             "they must also pay the tax. The result? The **exact same** "
             "equilibrium quantities and effective prices.\n\n"
             "Toggle the *Tax levied on* dropdown to see this for yourself."),
            ("What determines who bears the burden?",
             "The side of the market that is **more inelastic** (less responsive "
             "to price changes) bears **more** of the tax burden.\n\n"
             "- If demand is perfectly inelastic (vertical), buyers bear 100%\n"
             "- If supply is perfectly inelastic (vertical), sellers bear 100%\n"
             "- With equal elasticities, the burden is split 50/50\n\n"
             "Intuitively: the side that *can't easily walk away* gets stuck "
             "paying more of the tax."),
            ("What is deadweight loss?",
             "The tax reduces the quantity traded from Q* to Q_tax. The lost "
             "trades between Q_tax and Q* were mutually beneficial — buyers "
             "valued them above sellers' costs. The **deadweight loss** (DWL) "
             "is the total surplus destroyed by these lost trades, shown as "
             "the triangle between the supply and demand curves.\n\n"
             r"$$\text{DWL} = \frac{1}{2} \times (Q^* - Q_{\text{tax}}) "
             r"\times (P_{\text{buyer}} - P_{\text{seller}})$$"),
            ("Per-unit vs ad valorem taxes",
             "A **per-unit** tax adds a fixed dollar amount per unit sold "
             "(e.g. \\$0.50 per gallon of gas). The supply curve shifts up by "
             "a constant amount — a parallel shift.\n\n"
             "An **ad valorem** tax is a percentage of the price (e.g. a 20% "
             "sales tax). The tax wedge grows with the price, so the supply "
             "curve *rotates* rather than shifts. Ad valorem taxes generate "
             "more revenue on expensive goods and are the most common form of "
             "sales tax worldwide."),
        ]
        sections.append((
                "Elasticity formula for tax incidence (Advanced)",
                "With linear supply and demand, the buyer's share of a per-unit tax is:\n\n"
                r"$$\frac{\text{Buyer's burden}}{t} = \frac{E_s}{E_s + E_d} "
                r"= \frac{1/b_s}{1/b_s + 1/b_d} = \frac{b_d}{b_d + b_s}$$"
                "\n\nwhere b_d and b_s are the absolute slopes of demand and supply. "
                "The more inelastic side (steeper slope, smaller elasticity) bears "
                "a larger share.\n\n"
                "For ad valorem taxes, the same elasticity intuition holds but the "
                "algebra is slightly different because the wedge is multiplicative: "
                r"$P_b = P_s(1+\tau)$. The burden shares depend on the elasticities "
                "evaluated at the post-tax equilibrium, not just the slopes."
            ))
        return sections


# ---------------------------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------------------------

def main():
    concept = TaxIncidence()

    # --- Sidebar ---
    st.sidebar.header("Tax Incidence")
    st.sidebar.markdown("---")

    st.sidebar.subheader("Demand curve")
    a_d = st.sidebar.slider("Intercept (a)", 10.0, 50.0, 30.0, 1.0,
                            help="P = a - b·Q")
    b_d = st.sidebar.slider("Slope (b)", 0.1, 3.0, 1.0, 0.1,
                            help="Steeper = more inelastic demand")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Supply curve")
    a_s = st.sidebar.slider("Intercept (c)", 0.0, 20.0, 2.0, 1.0,
                            help="P = c + d·Q")
    b_s = st.sidebar.slider("Slope (d)", 0.1, 3.0, 1.0, 0.1,
                            help="Steeper = more inelastic supply")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Tax")
    tax_type = st.sidebar.selectbox("Tax type", ["Per-unit", "Ad valorem"])

    if tax_type == "Per-unit":
        tax = st.sidebar.slider("Per-unit tax (t)", 0.0, 20.0, 5.0, 0.5)
        tax_rate = 0.0
    else:
        tax = 0.0
        tax_rate_pct = st.sidebar.slider("Ad valorem rate (%)", 0, 100, 20, 1,
                                         help="Tax as a percentage of price")
        tax_rate = tax_rate_pct / 100.0

    statutory = st.sidebar.selectbox("Tax levied on", ["Sellers", "Buyers"])

    p = {
        "a_d": a_d, "b_d": b_d,
        "a_s": a_s, "b_s": b_s,
        "tax_type": tax_type,
        "tax": tax, "tax_rate": tax_rate,
        "statutory": statutory,
    }

    curves = concept.compute_curves(p)
    eq = concept.compute_equilibrium(p)

    # --- Main area ---
    st.title("Tax Incidence")
    st.markdown(concept.description)

    st.markdown("---")

    st.markdown("""
**How this app works:** The sidebar defines linear supply and demand curves
and a tax (either a fixed per-unit amount or an ad valorem percentage). The
app computes the pre-tax and post-tax equilibria and shows how the tax wedge
is split between buyers and sellers. Try making one side more inelastic
(steeper slope) to see the burden shift, or toggle *Tax levied on* between
Sellers and Buyers to confirm that statutory incidence doesn't affect the
economic outcome.
""")

    st.markdown("---")

    # Check for valid equilibrium
    if eq["Q_star"] <= 0:
        st.error("No equilibrium exists with these parameters (supply intercept "
                 "is above demand intercept).")
        return
    if eq["Q_tax"] <= 0 and (tax > 0 or tax_rate > 0):
        st.warning("The tax is large enough to eliminate all trade.")

    # Tax label
    if tax_type == "Per-unit":
        tax_label = f"${tax:.2f}/unit"
    else:
        tax_label = f"{tax_rate:.0%} ad valorem"

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(9, 6))
    q = curves["q"]

    # Original curves
    ax.plot(q, curves["demand"], color="#2563eb", linewidth=2, label="Demand")
    ax.plot(q, curves["supply"], color="#dc2626", linewidth=2, label="Supply")

    # Shifted curve (for visual)
    has_tax = (tax_type == "Per-unit" and tax > 0) or (tax_type == "Ad valorem" and tax_rate > 0)
    if has_tax:
        if tax_type == "Per-unit":
            if statutory == "Sellers":
                shifted = curves["supply"] + tax
                ax.plot(q, shifted, color="#dc2626", linewidth=1.5, linestyle="--",
                        label=f"Supply + tax ({tax_label})")
            else:
                shifted = curves["demand"] - tax
                ax.plot(q, shifted, color="#2563eb", linewidth=1.5, linestyle="--",
                        label=f"Demand − tax ({tax_label})")
        else:  # Ad valorem
            if statutory == "Sellers":
                # Sellers must remit τ of the price they receive, so gross price
                # buyers see is P_supply * (1 + τ)
                shifted = curves["supply"] * (1 + tax_rate)
                ax.plot(q, shifted, color="#dc2626", linewidth=1.5, linestyle="--",
                        label=f"Supply × (1+τ) ({tax_label})")
            else:
                # Buyers pay P_demand but only P_demand/(1+τ) goes to sellers
                shifted = curves["demand"] / (1 + tax_rate)
                ax.plot(q, shifted, color="#2563eb", linewidth=1.5, linestyle="--",
                        label=f"Demand / (1+τ) ({tax_label})")

    Q_s, P_s = eq["Q_star"], eq["P_star"]
    Q_t = eq["Q_tax"]
    P_b, P_sel = eq["P_buyer"], eq["P_seller"]

    # Pre-tax equilibrium dot
    ax.plot(Q_s, P_s, "o", color="#64748b", markersize=8, zorder=5,
            label=f"Pre-tax eq. (Q={Q_s:.1f}, P={P_s:.2f})")

    if has_tax and Q_t > 0:
        # Tax wedge: horizontal lines at P_buyer and P_seller
        ax.hlines(P_b, 0, Q_t, colors="#2563eb", linestyles=":", linewidth=1.2)
        ax.hlines(P_sel, 0, Q_t, colors="#dc2626", linestyles=":", linewidth=1.2)

        # Vertical tax wedge line
        ax.plot([Q_t, Q_t], [P_sel, P_b], color="#7c3aed", linewidth=2.5, zorder=4)
        wedge_label = f"t = ${eq['t_effective']:.2f}" if tax_type == "Per-unit" else f"τ·P = ${eq['t_effective']:.2f}"
        ax.text(Q_t + 0.3, (P_b + P_sel) / 2, wedge_label,
                color="#7c3aed", fontsize=10, fontweight="bold", va="center")

        # Buyer burden shading
        rect_buyer = mpatches.FancyBboxPatch(
            (0, P_s), Q_t, P_b - P_s,
            boxstyle="square,pad=0", facecolor="#2563eb", alpha=0.10,
            edgecolor="none")
        ax.add_patch(rect_buyer)

        # Seller burden shading
        rect_seller = mpatches.FancyBboxPatch(
            (0, P_sel), Q_t, P_s - P_sel,
            boxstyle="square,pad=0", facecolor="#dc2626", alpha=0.10,
            edgecolor="none")
        ax.add_patch(rect_seller)

        # DWL triangle
        ax.fill([Q_t, Q_s, Q_t], [P_b, P_s, P_sel],
                color="#f59e0b", alpha=0.25, label=f"DWL = ${eq['dwl']:.2f}")

        # Label buyer and seller prices
        ax.annotate(f"P_buyer = ${P_b:.2f}", xy=(Q_t, P_b),
                    xytext=(Q_t + 1.5, P_b + 1),
                    fontsize=9, color="#2563eb", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#2563eb", lw=1))
        ax.annotate(f"P_seller = ${P_sel:.2f}", xy=(Q_t, P_sel),
                    xytext=(Q_t + 1.5, P_sel - 1.5),
                    fontsize=9, color="#dc2626", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1))

        # Post-tax quantity line
        ax.axvline(Q_t, color="#94a3b8", linestyle=":", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("Quantity", fontsize=11)
    ax.set_ylabel("Price ($)", fontsize=11)
    ax.set_title("Supply, Demand, and Tax Wedge", fontsize=13, fontweight="bold")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")

    # Headline metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pre-tax price", f"${eq['P_star']:.2f}")
    c2.metric("Buyers pay", f"${eq['P_buyer']:.2f}",
              delta=f"+${eq['buyer_burden']:.2f}" if eq['buyer_burden'] > 0 else None)
    c3.metric("Sellers receive", f"${eq['P_seller']:.2f}",
              delta=f"-${eq['seller_burden']:.2f}" if eq['seller_burden'] > 0 else None,
              delta_color="inverse")
    c4.metric("Quantity", f"{eq['Q_tax']:.1f}",
              delta=f"{eq['Q_tax'] - eq['Q_star']:.1f} from {eq['Q_star']:.1f}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Buyer's share", f"{eq['buyer_share']:.0%}")
    c2.metric("Seller's share", f"{eq['seller_share']:.0%}")
    c3.metric("Tax revenue", f"${eq['tax_revenue']:.2f}")
    c4.metric("Deadweight loss", f"${eq['dwl']:.2f}")

    st.markdown("---")

    st.markdown("**Elasticities at pre-tax equilibrium**")
    c1, c2 = st.columns(2)
    c1.metric("Demand elasticity |Ed|", f"{eq['elasticity_d']:.2f}")
    c2.metric("Supply elasticity Es", f"{eq['elasticity_s']:.2f}")
    st.caption(
        "The more inelastic side (lower elasticity) bears a larger share of the "
        "tax. With equal slopes, the burden is split 50/50 regardless of "
        "statutory incidence."
    )

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
        page_title="Tax Incidence",
        page_icon="📊",
        layout="wide",
    )
    main()
