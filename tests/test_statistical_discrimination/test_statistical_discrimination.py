"""Tests for the Statistical Discrimination app."""

import numpy as np

from apps.statistical_discrimination.app import StatisticalDiscrimination


DEFAULT = {
    "mu_A": 15.0, "mu_B": 15.0, "sigma_theta": 4.0,
    "sigma_eps_A": 2.0, "sigma_eps_B": 6.0,
}


class TestStatisticalDiscrimination:
    def test_shrinkage_between_0_and_1(self):
        """Shrinkage factors must be in [0, 1]."""
        c = StatisticalDiscrimination()
        eq = c.compute_equilibrium(DEFAULT)
        assert 0 < eq["B_A"] < 1
        assert 0 < eq["B_B"] < 1

    def test_noisier_group_has_higher_shrinkage(self):
        """Group with noisier signal should have higher B (more profiling)."""
        c = StatisticalDiscrimination()
        eq = c.compute_equilibrium(DEFAULT)
        # σ_ε_B > σ_ε_A → B_B > B_A
        assert eq["B_B"] > eq["B_A"]

    def test_slope_plus_shrinkage_equals_one(self):
        """slope = 1 - B for each group."""
        c = StatisticalDiscrimination()
        eq = c.compute_equilibrium(DEFAULT)
        assert abs(eq["slope_A"] + eq["B_A"] - 1) < 1e-10
        assert abs(eq["slope_B"] + eq["B_B"] - 1) < 1e-10

    def test_wage_at_group_mean_equals_group_mean(self):
        """When signal = μ_g, wage offer should equal μ_g."""
        c = StatisticalDiscrimination()
        eq = c.compute_equilibrium(DEFAULT)
        w_A = c.wage_offer(DEFAULT["mu_A"], "A", eq)
        w_B = c.wage_offer(DEFAULT["mu_B"], "B", eq)
        assert abs(w_A - DEFAULT["mu_A"]) < 1e-10
        assert abs(w_B - DEFAULT["mu_B"]) < 1e-10

    def test_average_wage_equals_mean_productivity(self):
        """E[w] = μ for each group (Bayesian unbiasedness)."""
        c = StatisticalDiscrimination()
        eq = c.compute_equilibrium(DEFAULT)
        assert abs(eq["wage_mean_A"] - DEFAULT["mu_A"]) < 1e-10
        assert abs(eq["wage_mean_B"] - DEFAULT["mu_B"]) < 1e-10

    def test_wage_sd_less_than_productivity_sd(self):
        """Wage distribution should be compressed relative to true productivity."""
        c = StatisticalDiscrimination()
        eq = c.compute_equilibrium(DEFAULT)
        assert eq["wage_sd_A"] < DEFAULT["sigma_theta"]
        assert eq["wage_sd_B"] < DEFAULT["sigma_theta"]

    def test_noisier_group_more_compressed(self):
        """Noisier group should have more compressed wage distribution."""
        c = StatisticalDiscrimination()
        eq = c.compute_equilibrium(DEFAULT)
        assert eq["wage_sd_B"] < eq["wage_sd_A"]

    def test_no_discrimination_when_signals_equal(self):
        """With identical noise, wage functions should be identical (same means)."""
        c = StatisticalDiscrimination()
        p = {**DEFAULT, "sigma_eps_B": DEFAULT["sigma_eps_A"]}
        eq = c.compute_equilibrium(p)
        assert abs(eq["B_A"] - eq["B_B"]) < 1e-10
        assert abs(eq["slope_A"] - eq["slope_B"]) < 1e-10
        # Same signal → same wage
        w_A = c.wage_offer(20.0, "A", eq)
        w_B = c.wage_offer(20.0, "B", eq)
        assert abs(w_A - w_B) < 1e-10

    def test_perfect_signal_no_shrinkage(self):
        """With σ_ε → very small, B → 0 and wage → signal."""
        c = StatisticalDiscrimination()
        p = {**DEFAULT, "sigma_eps_A": 0.01, "sigma_eps_B": 0.01}
        eq = c.compute_equilibrium(p)
        assert eq["B_A"] < 0.001
        assert eq["B_B"] < 0.001
        # Wage should nearly equal signal
        s_test = 25.0
        assert abs(c.wage_offer(s_test, "A", eq) - s_test) < 0.1

    def test_very_noisy_signal_full_profiling(self):
        """With σ_ε → very large, B → 1 and wage → μ."""
        c = StatisticalDiscrimination()
        p = {**DEFAULT, "sigma_eps_B": 1000.0}
        eq = c.compute_equilibrium(p)
        assert eq["B_B"] > 0.999
        # Wage for any signal should be ≈ μ_B
        assert abs(c.wage_offer(100.0, "B", eq) - DEFAULT["mu_B"]) < 0.1

    def test_wage_gap_with_same_means_and_different_noise(self):
        """Same means: high-signal worker from noisy group is underpaid."""
        c = StatisticalDiscrimination()
        eq = c.compute_equilibrium(DEFAULT)
        # At a signal well above the mean
        s_high = DEFAULT["mu_A"] + 2 * DEFAULT["sigma_theta"]
        w_A = c.wage_offer(s_high, "A", eq)
        w_B = c.wage_offer(s_high, "B", eq)
        # Group A (less noisy) should pay higher for high signal
        assert w_A > w_B

    def test_wage_gap_reverses_below_mean(self):
        """Same means: low-signal worker from noisy group is overpaid."""
        c = StatisticalDiscrimination()
        eq = c.compute_equilibrium(DEFAULT)
        s_low = DEFAULT["mu_A"] - 2 * DEFAULT["sigma_theta"]
        w_A = c.wage_offer(s_low, "A", eq)
        w_B = c.wage_offer(s_low, "B", eq)
        assert w_B > w_A

    def test_crossing_point_exists_with_same_means(self):
        """With same means and different noise, wage lines cross at s = μ."""
        c = StatisticalDiscrimination()
        eq = c.compute_equilibrium(DEFAULT)
        assert eq["s_equal"] is not None
        # Should cross at the common mean
        assert abs(eq["s_equal"] - DEFAULT["mu_A"]) < 1e-10

    def test_different_means_shift_crossing(self):
        """With different means, crossing point shifts."""
        c = StatisticalDiscrimination()
        p = {**DEFAULT, "mu_B": 12.0}
        eq = c.compute_equilibrium(p)
        assert eq["s_equal"] is not None
        # Crossing should not be at either mean
        assert eq["s_equal"] != p["mu_A"]

    def test_shrinkage_formula(self):
        """B = σ²_ε / (σ²_θ + σ²_ε)."""
        c = StatisticalDiscrimination()
        eq = c.compute_equilibrium(DEFAULT)
        expected_A = DEFAULT["sigma_eps_A"] ** 2 / (
            DEFAULT["sigma_theta"] ** 2 + DEFAULT["sigma_eps_A"] ** 2)
        expected_B = DEFAULT["sigma_eps_B"] ** 2 / (
            DEFAULT["sigma_theta"] ** 2 + DEFAULT["sigma_eps_B"] ** 2)
        assert abs(eq["B_A"] - expected_A) < 1e-10
        assert abs(eq["B_B"] - expected_B) < 1e-10

    def test_posterior_variance_formula(self):
        """Posterior var = σ²_θ · σ²_ε / (σ²_θ + σ²_ε)."""
        c = StatisticalDiscrimination()
        eq = c.compute_equilibrium(DEFAULT)
        var_t = DEFAULT["sigma_theta"] ** 2
        expected_A = var_t * DEFAULT["sigma_eps_A"] ** 2 / (
            var_t + DEFAULT["sigma_eps_A"] ** 2)
        expected_B = var_t * DEFAULT["sigma_eps_B"] ** 2 / (
            var_t + DEFAULT["sigma_eps_B"] ** 2)
        assert abs(eq["post_var_A"] - expected_A) < 1e-10
        assert abs(eq["post_var_B"] - expected_B) < 1e-10
