"""Tests for shared analytic power utilities."""

from shared.stats_utils import power_from_noncentral_t, power_from_normal_approx


class TestNoncentralT:
    def test_zero_ncp_gives_alpha(self):
        power = power_from_noncentral_t(ncp=0, df=29, alpha=0.05)
        assert abs(power - 0.05) < 0.001

    def test_large_ncp_gives_high_power(self):
        power = power_from_noncentral_t(ncp=5.0, df=29, alpha=0.05)
        assert power > 0.99

    def test_power_increases_with_ncp(self):
        p1 = power_from_noncentral_t(ncp=1.0, df=29, alpha=0.05)
        p2 = power_from_noncentral_t(ncp=2.0, df=29, alpha=0.05)
        assert p2 > p1


class TestNormalApprox:
    def test_zero_ncp_gives_alpha(self):
        power = power_from_normal_approx(z_ncp=0, alpha=0.05)
        assert abs(power - 0.05) < 0.001

    def test_large_ncp_gives_high_power(self):
        power = power_from_normal_approx(z_ncp=5.0, alpha=0.05)
        assert power > 0.99
