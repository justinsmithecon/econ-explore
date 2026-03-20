"""Tests for the Spence Signalling Model app."""

import numpy as np
import pytest

from apps.signalling.app import SpenceSignalling


DEFAULT = {
    "theta_H": 20.0, "theta_L": 10.0, "c_H": 2.0, "c_L": 5.0,
    "lam": 0.4, "e_star_frac": 0.5,
}


class TestSpenceSignalling:
    def test_e_star_in_feasible_range(self):
        """e* should lie between e_min and e_max."""
        c = SpenceSignalling()
        eq = c.compute_equilibrium(DEFAULT)
        assert eq["e_min"] <= eq["e_star"] <= eq["e_max"]

    def test_e_min_formula(self):
        """e_min = (θ_H - θ_L) / c_L."""
        c = SpenceSignalling()
        eq = c.compute_equilibrium(DEFAULT)
        expected = (DEFAULT["theta_H"] - DEFAULT["theta_L"]) / DEFAULT["c_L"]
        assert abs(eq["e_min"] - expected) < 1e-10

    def test_e_max_formula(self):
        """e_max = (θ_H - θ_L) / c_H."""
        c = SpenceSignalling()
        eq = c.compute_equilibrium(DEFAULT)
        expected = (DEFAULT["theta_H"] - DEFAULT["theta_L"]) / DEFAULT["c_H"]
        assert abs(eq["e_max"] - expected) < 1e-10

    def test_ic_H_satisfied(self):
        """H type should prefer signalling: θ_H - c_H·e* ≥ θ_L."""
        c = SpenceSignalling()
        eq = c.compute_equilibrium(DEFAULT)
        assert eq["ic_H_slack"] >= -1e-10

    def test_ic_L_satisfied(self):
        """L type should prefer not signalling: θ_H - c_L·e* ≤ θ_L."""
        c = SpenceSignalling()
        eq = c.compute_equilibrium(DEFAULT)
        assert eq["ic_L_slack"] <= 1e-10

    def test_ic_holds_at_e_min(self):
        """At e_min, both ICs should be satisfied (L is just indifferent)."""
        c = SpenceSignalling()
        eq = c.compute_equilibrium({**DEFAULT, "e_star_frac": 0.0})
        assert eq["ic_H_slack"] >= -1e-10
        assert abs(eq["ic_L_slack"]) < 1e-10  # L indifferent at e_min

    def test_ic_holds_at_e_max(self):
        """At e_max, both ICs should be satisfied (H is just indifferent)."""
        c = SpenceSignalling()
        eq = c.compute_equilibrium({**DEFAULT, "e_star_frac": 1.0})
        assert abs(eq["ic_H_slack"]) < 1e-10  # H indifferent at e_max
        assert eq["ic_L_slack"] <= 1e-10

    def test_sep_payoff_H_equals_theta_H_minus_cost(self):
        """H type's net payoff = θ_H - c_H · e*."""
        c = SpenceSignalling()
        eq = c.compute_equilibrium(DEFAULT)
        expected = DEFAULT["theta_H"] - DEFAULT["c_H"] * eq["e_star"]
        assert abs(eq["sep_payoff_H"] - expected) < 1e-10

    def test_sep_payoff_L_equals_theta_L(self):
        """L type gets θ_L in separating equilibrium (no education cost)."""
        c = SpenceSignalling()
        eq = c.compute_equilibrium(DEFAULT)
        assert abs(eq["sep_payoff_L"] - DEFAULT["theta_L"]) < 1e-10

    def test_pooling_wage_formula(self):
        """Pooling wage = λ·θ_H + (1-λ)·θ_L."""
        c = SpenceSignalling()
        eq = c.compute_equilibrium(DEFAULT)
        expected = DEFAULT["lam"] * DEFAULT["theta_H"] + (1 - DEFAULT["lam"]) * DEFAULT["theta_L"]
        assert abs(eq["pool_wage"] - expected) < 1e-10

    def test_pooling_surplus_equals_first_best(self):
        """In this model, pooling and first-best have the same total surplus."""
        c = SpenceSignalling()
        eq = c.compute_equilibrium(DEFAULT)
        assert abs(eq["pool_surplus"] - eq["fb_surplus"]) < 1e-10

    def test_separating_surplus_less_than_pooling(self):
        """Signalling wastes resources, so separating surplus < pooling."""
        c = SpenceSignalling()
        eq = c.compute_equilibrium(DEFAULT)
        assert eq["sep_surplus"] < eq["pool_surplus"]

    def test_signal_cost_positive(self):
        """With e* > 0, there is positive signalling cost."""
        c = SpenceSignalling()
        eq = c.compute_equilibrium(DEFAULT)
        assert eq["signal_cost"] > 0

    def test_surplus_decomposition(self):
        """Separating surplus = pooling surplus - signalling cost."""
        c = SpenceSignalling()
        eq = c.compute_equilibrium(DEFAULT)
        expected = eq["pool_surplus"] - eq["signal_cost"]
        assert abs(eq["sep_surplus"] - expected) < 1e-10

    def test_H_prefers_separating_over_pooling_when_lambda_low(self):
        """When λ is low, pooling wage is close to θ_L, so H prefers separating."""
        c = SpenceSignalling()
        eq = c.compute_equilibrium({**DEFAULT, "lam": 0.2, "e_star_frac": 0.0})
        assert eq["sep_payoff_H"] > eq["pool_payoff_H"]
        assert not eq["pareto_dominated"]
        assert not eq["h_prefers_pooling"]

    def test_L_prefers_pooling_over_separating(self):
        """L types always prefer pooling (average wage > θ_L when λ > 0)."""
        c = SpenceSignalling()
        eq = c.compute_equilibrium(DEFAULT)
        assert eq["pool_payoff_L"] > eq["sep_payoff_L"]

    def test_pareto_dominated_with_high_lambda_and_high_e_star(self):
        """When λ is high and e* is costly, pooling Pareto-dominates."""
        c = SpenceSignalling()
        # High λ → pooling wage is high; high e* → signalling is costly
        eq = c.compute_equilibrium({**DEFAULT, "lam": 0.5, "e_star_frac": 1.0})
        assert eq["pareto_dominated"]
        assert eq["h_prefers_pooling"]
        assert eq["pool_payoff_H"] > eq["sep_payoff_H"]

    def test_not_pareto_dominated_with_low_lambda(self):
        """When λ is very low and e* is minimal, H types prefer separating."""
        c = SpenceSignalling()
        eq = c.compute_equilibrium({**DEFAULT, "lam": 0.1, "e_star_frac": 0.0})
        assert not eq["pareto_dominated"]
        assert not eq["h_prefers_pooling"]

    def test_frac_0_gives_e_min(self):
        """e_star_frac = 0 should give e* = e_min."""
        c = SpenceSignalling()
        eq = c.compute_equilibrium({**DEFAULT, "e_star_frac": 0.0})
        assert abs(eq["e_star"] - eq["e_min"]) < 1e-10

    def test_frac_1_gives_e_max(self):
        """e_star_frac = 1 should give e* = e_max."""
        c = SpenceSignalling()
        eq = c.compute_equilibrium({**DEFAULT, "e_star_frac": 1.0})
        assert abs(eq["e_star"] - eq["e_max"]) < 1e-10
