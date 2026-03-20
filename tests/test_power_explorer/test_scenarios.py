"""Tests for power explorer scenarios."""

from apps.power_explorer.scenarios.one_sample_mean import OneSampleMean
from apps.power_explorer.scenarios.two_sample_mean import TwoSampleMean
from apps.power_explorer.scenarios.proportion import ProportionTest
from apps.power_explorer.scenarios.regression_slope import RegressionSlope


class TestOneSampleMean:
    def test_known_power(self):
        s = OneSampleMean()
        power = s.analytic_power({
            "mu_0": 0, "mu_1": 10, "sigma": 5, "n": 100, "alpha": 0.05
        })
        assert power > 0.99

    def test_no_effect_gives_alpha(self):
        s = OneSampleMean()
        power = s.analytic_power({
            "mu_0": 50, "mu_1": 50, "sigma": 10, "n": 30, "alpha": 0.05
        })
        assert abs(power - 0.05) < 0.001

    def test_power_increases_with_n(self):
        s = OneSampleMean()
        params = {"mu_0": 50, "mu_1": 55, "sigma": 10, "alpha": 0.05}
        p1 = s.analytic_power({**params, "n": 20})
        p2 = s.analytic_power({**params, "n": 100})
        assert p2 > p1


class TestTwoSampleMean:
    def test_no_effect(self):
        s = TwoSampleMean()
        power = s.analytic_power({
            "mu_1": 50, "mu_2": 50, "sigma": 10, "n_per_group": 30, "alpha": 0.05
        })
        assert abs(power - 0.05) < 0.001

    def test_large_effect(self):
        s = TwoSampleMean()
        power = s.analytic_power({
            "mu_1": 50, "mu_2": 60, "sigma": 5, "n_per_group": 100, "alpha": 0.05
        })
        assert power > 0.99


class TestProportion:
    def test_no_effect(self):
        s = ProportionTest()
        power = s.analytic_power({
            "p_0": 0.5, "p_1": 0.5, "n": 100, "alpha": 0.05
        })
        assert abs(power - 0.05) < 0.001


class TestRegressionSlope:
    def test_no_effect(self):
        s = RegressionSlope()
        power = s.analytic_power({
            "beta": 0, "sigma_x": 2, "sigma_eps": 5, "n": 50, "alpha": 0.05
        })
        assert abs(power - 0.05) < 0.001

    def test_large_effect(self):
        s = RegressionSlope()
        power = s.analytic_power({
            "beta": 2, "sigma_x": 5, "sigma_eps": 1, "n": 100, "alpha": 0.05
        })
        assert power > 0.99
