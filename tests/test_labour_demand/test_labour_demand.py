"""Tests for the Labour Demand decomposition app."""

import numpy as np

from apps.labour_demand.app import LabourDemand


DEFAULT = {
    "alpha": 0.5, "w_0": 5.0, "w_1": 8.0, "r": 5.0,
    "P_bar": 50.0, "delta": 1.0,
}


class TestLabourDemand:
    def test_decomposition_sums_to_total(self):
        """Substitution + scale = total effect."""
        c = LabourDemand()
        eq = c.compute_equilibrium(DEFAULT)
        total = eq["substitution_effect"] + eq["scale_effect"]
        assert abs(total - eq["total_effect"]) < 1e-10

    def test_wage_increase_reduces_labour(self):
        """Higher wage should reduce total labour demand."""
        c = LabourDemand()
        eq = c.compute_equilibrium(DEFAULT)
        assert eq["total_effect"] < 0

    def test_substitution_effect_negative_for_wage_increase(self):
        """Substitution effect is always negative when w rises."""
        c = LabourDemand()
        eq = c.compute_equilibrium(DEFAULT)
        assert eq["substitution_effect"] < 0

    def test_scale_effect_negative_for_wage_increase(self):
        """Scale effect is negative when w rises (costs up, output down)."""
        c = LabourDemand()
        eq = c.compute_equilibrium(DEFAULT)
        assert eq["scale_effect"] < 0

    def test_no_change_when_wages_equal(self):
        """If w₁ = w₀, no effect."""
        c = LabourDemand()
        eq = c.compute_equilibrium({**DEFAULT, "w_1": DEFAULT["w_0"]})
        assert abs(eq["total_effect"]) < 1e-10
        assert abs(eq["substitution_effect"]) < 1e-10
        assert abs(eq["scale_effect"]) < 1e-10

    def test_substitution_keeps_output_constant(self):
        """At point B, output should still equal Y₀."""
        c = LabourDemand()
        eq = c.compute_equilibrium(DEFAULT)
        alpha = DEFAULT["alpha"]
        Y_at_B = eq["L_sub"] ** alpha * eq["K_sub"] ** (1 - alpha)
        assert abs(Y_at_B - eq["Y_0"]) < 1e-8

    def test_wage_decrease_increases_labour(self):
        """Lower wage should increase labour demand."""
        c = LabourDemand()
        eq = c.compute_equilibrium({**DEFAULT, "w_0": 8.0, "w_1": 5.0})
        assert eq["total_effect"] > 0
        assert eq["substitution_effect"] > 0
        assert eq["scale_effect"] > 0

    def test_higher_alpha_more_labour_intensive(self):
        """Higher α means more labour-intensive production."""
        c = LabourDemand()
        eq_low = c.compute_equilibrium({**DEFAULT, "alpha": 0.3, "w_1": DEFAULT["w_0"]})
        eq_high = c.compute_equilibrium({**DEFAULT, "alpha": 0.7, "w_1": DEFAULT["w_0"]})
        assert eq_high["L_0"] > eq_low["L_0"]

    def test_cost_minimisation_foc(self):
        """At optimum, MPL/MPK = w/r."""
        c = LabourDemand()
        eq = c.compute_equilibrium(DEFAULT)
        alpha = DEFAULT["alpha"]
        L, K = eq["L_0"], eq["K_0"]
        MPL = alpha * (K / L) ** (1 - alpha)
        MPK = (1 - alpha) * (L / K) ** alpha
        assert abs(MPL / MPK - DEFAULT["w_0"] / DEFAULT["r"]) < 1e-8
