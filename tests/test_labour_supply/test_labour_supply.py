"""Tests for the Labour Supply income/substitution effects app."""

import numpy as np

from apps.labour_supply.app import LabourSupply


DEFAULT = {
    "beta": 0.5, "T": 16.0, "w_0": 10.0, "w_1": 15.0, "V": 20.0,
}


class TestLabourSupply:
    def test_decomposition_sums_to_total(self):
        """Substitution + income = total effect."""
        c = LabourSupply()
        eq = c.compute_equilibrium(DEFAULT)
        total = eq["substitution_effect"] + eq["income_effect"]
        assert abs(total - eq["total_effect"]) < 1e-10

    def test_substitution_positive_for_wage_increase(self):
        """Substitution effect always increases hours when wage rises."""
        c = LabourSupply()
        eq = c.compute_equilibrium(DEFAULT)
        assert eq["substitution_effect"] > 0

    def test_income_negative_for_wage_increase(self):
        """Income effect reduces hours when wage rises (leisure is normal)."""
        c = LabourSupply()
        eq = c.compute_equilibrium(DEFAULT)
        assert eq["income_effect"] < 0

    def test_no_change_when_wages_equal(self):
        """If w₁ = w₀, no effect."""
        c = LabourSupply()
        eq = c.compute_equilibrium({**DEFAULT, "w_1": DEFAULT["w_0"]})
        assert abs(eq["total_effect"]) < 1e-10
        assert abs(eq["substitution_effect"]) < 1e-10
        assert abs(eq["income_effect"]) < 1e-10

    def test_compensated_utility_equals_initial(self):
        """At point B, utility should equal U₀."""
        c = LabourSupply()
        eq = c.compute_equilibrium(DEFAULT)
        beta = DEFAULT["beta"]
        U_at_B = eq["C_comp"] ** beta * eq["ell_comp"] ** (1 - beta)
        assert abs(U_at_B - eq["U_0"]) < 1e-8

    def test_compensated_tangency(self):
        """At point B, MRS = w₁ (tangent to new wage)."""
        c = LabourSupply()
        eq = c.compute_equilibrium(DEFAULT)
        beta = DEFAULT["beta"]
        # MRS = (1-β)C / (β·ℓ) should equal w₁
        mrs = (1 - beta) * eq["C_comp"] / (beta * eq["ell_comp"])
        assert abs(mrs - DEFAULT["w_1"]) < 1e-8

    def test_budget_constraint_holds(self):
        """Initial and final points should satisfy their budget constraints."""
        c = LabourSupply()
        eq = c.compute_equilibrium(DEFAULT)
        T, V = DEFAULT["T"], DEFAULT["V"]
        w_0, w_1 = DEFAULT["w_0"], DEFAULT["w_1"]
        # C = w(T - ℓ) + V
        assert abs(eq["C_0"] - (w_0 * (T - eq["ell_0"]) + V)) < 1e-8
        assert abs(eq["C_1"] - (w_1 * (T - eq["ell_1"]) + V)) < 1e-8

    def test_hours_plus_leisure_equals_T(self):
        """h + ℓ = T at every point."""
        c = LabourSupply()
        eq = c.compute_equilibrium(DEFAULT)
        T = DEFAULT["T"]
        assert abs(eq["h_0"] + eq["ell_0"] - T) < 1e-10
        assert abs(eq["h_comp"] + eq["ell_comp"] - T) < 1e-10
        assert abs(eq["h_1"] + eq["ell_1"] - T) < 1e-10

    def test_zero_nonlabour_income_constant_hours(self):
        """With V=0 and Cobb-Douglas, h* = βT regardless of wage."""
        c = LabourSupply()
        p = {**DEFAULT, "V": 0.0, "w_0": 5.0, "w_1": 20.0}
        eq = c.compute_equilibrium(p)
        beta, T = p["beta"], p["T"]
        assert abs(eq["h_0"] - beta * T) < 1e-10
        assert abs(eq["h_1"] - beta * T) < 1e-10
        # Total effect should be zero
        assert abs(eq["total_effect"]) < 1e-10

    def test_wage_decrease_effects_reverse(self):
        """When wage falls, substitution reduces hours, income increases them."""
        c = LabourSupply()
        eq = c.compute_equilibrium({**DEFAULT, "w_0": 15.0, "w_1": 10.0})
        assert eq["substitution_effect"] < 0
        assert eq["income_effect"] > 0

    def test_labour_supply_curve_shape(self):
        """Supply curve returns valid arrays."""
        c = LabourSupply()
        wages, hours = c.labour_supply_curve(0.5, 16.0, 20.0)
        assert len(wages) == len(hours)
        assert len(wages) > 0
        # Hours should be finite
        assert np.all(np.isfinite(hours))
