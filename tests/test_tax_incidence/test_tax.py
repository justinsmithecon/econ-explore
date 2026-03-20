"""Tests for the Tax Incidence app."""

import numpy as np

from apps.tax_incidence.app import TaxIncidence


BASE_PERUNIT = {
    "a_d": 30.0, "b_d": 1.0,
    "a_s": 2.0, "b_s": 1.0,
    "tax_type": "Per-unit",
    "tax": 5.0, "tax_rate": 0.0,
    "statutory": "Sellers",
}

BASE_ADVAL = {
    "a_d": 30.0, "b_d": 1.0,
    "a_s": 2.0, "b_s": 1.0,
    "tax_type": "Ad valorem",
    "tax": 0.0, "tax_rate": 0.20,
    "statutory": "Sellers",
}


class TestPerUnitTax:
    def test_pretax_equilibrium(self):
        c = TaxIncidence()
        eq = c.compute_equilibrium({**BASE_PERUNIT, "tax": 0})
        assert abs(eq["Q_star"] - 14.0) < 1e-10
        assert abs(eq["P_star"] - 16.0) < 1e-10

    def test_tax_wedge_equals_tax(self):
        c = TaxIncidence()
        eq = c.compute_equilibrium(BASE_PERUNIT)
        assert abs((eq["P_buyer"] - eq["P_seller"]) - BASE_PERUNIT["tax"]) < 1e-10

    def test_burden_sums_to_tax(self):
        c = TaxIncidence()
        eq = c.compute_equilibrium(BASE_PERUNIT)
        assert abs(eq["buyer_burden"] + eq["seller_burden"] - BASE_PERUNIT["tax"]) < 1e-10

    def test_equal_slopes_equal_burden(self):
        c = TaxIncidence()
        eq = c.compute_equilibrium(BASE_PERUNIT)
        assert abs(eq["buyer_share"] - 0.5) < 1e-10

    def test_inelastic_demand_bears_more(self):
        c = TaxIncidence()
        eq = c.compute_equilibrium({**BASE_PERUNIT, "b_d": 2.0, "b_s": 1.0})
        assert abs(eq["buyer_share"] - 2.0 / 3.0) < 1e-10

    def test_inelastic_supply_bears_more(self):
        c = TaxIncidence()
        eq = c.compute_equilibrium({**BASE_PERUNIT, "b_d": 1.0, "b_s": 2.0})
        assert abs(eq["seller_share"] - 2.0 / 3.0) < 1e-10

    def test_statutory_incidence_irrelevant(self):
        c = TaxIncidence()
        eq_s = c.compute_equilibrium({**BASE_PERUNIT, "statutory": "Sellers"})
        eq_b = c.compute_equilibrium({**BASE_PERUNIT, "statutory": "Buyers"})
        assert abs(eq_s["P_buyer"] - eq_b["P_buyer"]) < 1e-10
        assert abs(eq_s["P_seller"] - eq_b["P_seller"]) < 1e-10

    def test_no_tax_no_burden(self):
        c = TaxIncidence()
        eq = c.compute_equilibrium({**BASE_PERUNIT, "tax": 0})
        assert abs(eq["buyer_burden"]) < 1e-10
        assert abs(eq["seller_burden"]) < 1e-10
        assert abs(eq["dwl"]) < 1e-10

    def test_dwl_positive_with_tax(self):
        c = TaxIncidence()
        eq = c.compute_equilibrium(BASE_PERUNIT)
        assert eq["dwl"] > 0

    def test_dwl_formula(self):
        c = TaxIncidence()
        eq = c.compute_equilibrium(BASE_PERUNIT)
        expected = 0.5 * BASE_PERUNIT["tax"] * (eq["Q_star"] - eq["Q_tax"])
        assert abs(eq["dwl"] - expected) < 1e-10

    def test_tax_revenue(self):
        c = TaxIncidence()
        eq = c.compute_equilibrium(BASE_PERUNIT)
        assert abs(eq["tax_revenue"] - BASE_PERUNIT["tax"] * eq["Q_tax"]) < 1e-10


class TestAdValoremTax:
    def test_wedge_equals_tau_times_p_seller(self):
        """P_buyer - P_seller = τ * P_seller."""
        c = TaxIncidence()
        eq = c.compute_equilibrium(BASE_ADVAL)
        tau = BASE_ADVAL["tax_rate"]
        assert abs(eq["P_buyer"] - eq["P_seller"] * (1 + tau)) < 1e-10

    def test_burden_sums_to_wedge(self):
        c = TaxIncidence()
        eq = c.compute_equilibrium(BASE_ADVAL)
        assert abs(eq["buyer_burden"] + eq["seller_burden"] - eq["t_effective"]) < 1e-10

    def test_revenue_equals_tau_times_p_seller_times_q(self):
        c = TaxIncidence()
        eq = c.compute_equilibrium(BASE_ADVAL)
        tau = BASE_ADVAL["tax_rate"]
        expected = tau * eq["P_seller"] * eq["Q_tax"]
        assert abs(eq["tax_revenue"] - expected) < 1e-10

    def test_dwl_positive_with_tax(self):
        c = TaxIncidence()
        eq = c.compute_equilibrium(BASE_ADVAL)
        assert eq["dwl"] > 0

    def test_no_tax_no_effect(self):
        c = TaxIncidence()
        eq = c.compute_equilibrium({**BASE_ADVAL, "tax_rate": 0})
        assert abs(eq["buyer_burden"]) < 1e-10
        assert abs(eq["seller_burden"]) < 1e-10
        assert abs(eq["dwl"]) < 1e-10

    def test_statutory_incidence_irrelevant(self):
        c = TaxIncidence()
        eq_s = c.compute_equilibrium({**BASE_ADVAL, "statutory": "Sellers"})
        eq_b = c.compute_equilibrium({**BASE_ADVAL, "statutory": "Buyers"})
        assert abs(eq_s["P_buyer"] - eq_b["P_buyer"]) < 1e-10
        assert abs(eq_s["Q_tax"] - eq_b["Q_tax"]) < 1e-10

    def test_higher_rate_more_dwl(self):
        c = TaxIncidence()
        eq_low = c.compute_equilibrium({**BASE_ADVAL, "tax_rate": 0.10})
        eq_high = c.compute_equilibrium({**BASE_ADVAL, "tax_rate": 0.30})
        assert eq_high["dwl"] > eq_low["dwl"]

    def test_dwl_formula(self):
        """DWL = 0.5 * (Q_star - Q_tax) * (P_buyer - P_seller)."""
        c = TaxIncidence()
        eq = c.compute_equilibrium(BASE_ADVAL)
        expected = 0.5 * (eq["Q_star"] - eq["Q_tax"]) * (eq["P_buyer"] - eq["P_seller"])
        assert abs(eq["dwl"] - expected) < 1e-10
