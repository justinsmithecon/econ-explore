"""Tests for the Ability Bias app."""

import numpy as np
import pytest

from apps.ability_bias.app import AbilityBias


DEFAULT = {
    "beta_1": 1.5, "beta_2": 2.0, "rho": 0.6, "sigma": 3.0, "n": 500.0,
}

# Large sample for convergence tests
LARGE = {**DEFAULT, "n": 50_000.0, "sigma": 1.0}


class TestAbilityBias:
    def test_ovb_formula_holds_in_sample(self):
        """bias = β̂₂ · δ̂₁ should equal α̂₁ − β̂₁ exactly."""
        c = AbilityBias()
        rng = np.random.default_rng(42)
        data = c.generate_data(DEFAULT, rng)
        result = c.estimate(data, DEFAULT)
        assert abs(result["bias_formula"] - result["bias_estimated"]) < 1e-10

    def test_no_bias_when_rho_zero(self):
        """With ρ = 0, the naive estimate should be close to the true β₁."""
        c = AbilityBias()
        p = {**LARGE, "rho": 0.0}
        rng = np.random.default_rng(99)
        data = c.generate_data(p, rng)
        result = c.estimate(data, p)
        assert abs(result["bias_estimated"]) < 0.05

    def test_no_bias_when_beta2_zero(self):
        """If ability doesn't affect wages, no OVB even with ρ > 0."""
        c = AbilityBias()
        p = {**LARGE, "beta_2": 0.0}
        rng = np.random.default_rng(99)
        data = c.generate_data(p, rng)
        result = c.estimate(data, p)
        assert abs(result["bias_estimated"]) < 0.05

    def test_positive_bias_when_rho_positive(self):
        """With ρ > 0 and β₂ > 0, the naive estimate should exceed the true."""
        c = AbilityBias()
        rng = np.random.default_rng(42)
        data = c.generate_data(LARGE, rng)
        result = c.estimate(data, LARGE)
        assert result["beta_short"][1] > result["beta_long"][1] + 0.1

    def test_negative_bias_when_rho_negative(self):
        """With ρ < 0 and β₂ > 0, the naive estimate should be below the true."""
        c = AbilityBias()
        p = {**LARGE, "rho": -0.4}
        rng = np.random.default_rng(42)
        data = c.generate_data(p, rng)
        result = c.estimate(data, p)
        assert result["beta_short"][1] < result["beta_long"][1] - 0.1

    def test_long_regression_recovers_true_params(self):
        """With large n and low noise, the long regression should nail β₁ and β₂."""
        c = AbilityBias()
        rng = np.random.default_rng(42)
        data = c.generate_data(LARGE, rng)
        result = c.estimate(data, LARGE)
        assert abs(result["beta_long"][1] - LARGE["beta_1"]) < 0.05
        assert abs(result["beta_long"][2] - LARGE["beta_2"]) < 0.05

    def test_bias_converges_to_theoretical(self):
        """In large samples, the estimated bias should approach β₂ · ρ / σ_S."""
        c = AbilityBias()
        rng = np.random.default_rng(42)
        data = c.generate_data(LARGE, rng)
        result = c.estimate(data, LARGE)
        assert abs(result["bias_estimated"] - result["bias_theoretical"]) < 0.1

    def test_schooling_ability_correlation(self):
        """Generated data should have approximately the requested correlation."""
        c = AbilityBias()
        rng = np.random.default_rng(42)
        data = c.generate_data(LARGE, rng)
        r = np.corrcoef(data["schooling"], data["ability"])[0, 1]
        assert abs(r - LARGE["rho"]) < 0.02

    def test_different_seeds_give_different_data(self):
        """Unfixed seed should produce different samples."""
        c = AbilityBias()
        data1 = c.generate_data(DEFAULT, np.random.default_rng(1))
        data2 = c.generate_data(DEFAULT, np.random.default_rng(2))
        assert not np.allclose(data1["wage"], data2["wage"])

    def test_data_shapes(self):
        """All arrays should have the right length."""
        c = AbilityBias()
        rng = np.random.default_rng(42)
        data = c.generate_data(DEFAULT, rng)
        n = int(DEFAULT["n"])
        assert len(data["schooling"]) == n
        assert len(data["ability"]) == n
        assert len(data["wage"]) == n
