"""Tests for the Oaxaca-Blinder decomposition."""

import numpy as np

from apps.oaxaca_blinder.app import OaxacaBlinder


DEFAULT_PARAMS = {
    "n": 500,
    "edu_mean_a": 14.0, "edu_mean_b": 13.0,
    "intercept_a": 8.0, "beta_edu_a": 1.2,
    "intercept_b": 7.0, "beta_edu_b": 0.9,
    "sigma": 3.0,
}


class TestOaxacaBlinder:
    def test_decomposition_sums_to_total_gap(self):
        """Explained + unexplained must equal the total gap."""
        concept = OaxacaBlinder()
        rng = np.random.default_rng(42)
        data = concept.generate_data(DEFAULT_PARAMS, rng)
        result = concept.estimate(data, DEFAULT_PARAMS)

        total = result["total_gap"]
        decomp_sum = result["explained"] + result["unexplained"]
        assert abs(total - decomp_sum) < 1e-10

    def test_no_gap_when_groups_identical(self):
        """If both groups have the same DGP, the gap should be near zero."""
        concept = OaxacaBlinder()
        symmetric = {
            "n": 2000,
            "edu_mean_a": 14.0, "edu_mean_b": 14.0,
            "intercept_a": 8.0, "beta_edu_a": 1.0,
            "intercept_b": 8.0, "beta_edu_b": 1.0,
            "sigma": 2.0,
        }
        rng = np.random.default_rng(42)
        data = concept.generate_data(symmetric, rng)
        result = concept.estimate(data, symmetric)

        assert abs(result["total_gap"]) < 1.0
        assert abs(result["explained"]) < 0.5
        assert abs(result["unexplained"]) < 0.5

    def test_pure_discrimination_all_unexplained(self):
        """If characteristics are identical but returns differ, gap is all unexplained."""
        concept = OaxacaBlinder()
        params = {
            "n": 5000,
            "edu_mean_a": 14.0, "edu_mean_b": 14.0,
            "intercept_a": 10.0, "beta_edu_a": 1.5,
            "intercept_b": 8.0, "beta_edu_b": 1.0,
            "sigma": 1.0,
        }
        rng = np.random.default_rng(42)
        data = concept.generate_data(params, rng)
        result = concept.estimate(data, params)

        assert abs(result["explained"]) < 0.5
        assert result["unexplained"] > 0
        assert abs(result["total_gap"] - result["unexplained"]) < 0.5

    def test_pure_characteristics_all_explained(self):
        """If returns are identical but characteristics differ, gap is all explained."""
        concept = OaxacaBlinder()
        params = {
            "n": 5000,
            "edu_mean_a": 16.0, "edu_mean_b": 12.0,
            "intercept_a": 8.0, "beta_edu_a": 1.0,
            "intercept_b": 8.0, "beta_edu_b": 1.0,
            "sigma": 1.0,
        }
        rng = np.random.default_rng(42)
        data = concept.generate_data(params, rng)
        result = concept.estimate(data, params)

        assert abs(result["unexplained"]) < 0.5
        assert result["explained"] > 0
        assert abs(result["total_gap"] - result["explained"]) < 0.5

    def test_detail_sums_to_aggregate(self):
        """Per-variable explained/unexplained must sum to the aggregates."""
        concept = OaxacaBlinder()
        rng = np.random.default_rng(42)
        data = concept.generate_data(DEFAULT_PARAMS, rng)
        result = concept.estimate(data, DEFAULT_PARAMS)

        assert abs(result["explained_detail"].sum() - result["explained"]) < 1e-10
        assert abs(result["unexplained_detail"].sum() - result["unexplained"]) < 1e-10

    def test_ols_recovers_true_coefficients(self):
        """With low noise and large n, OLS should recover true coefficients."""
        concept = OaxacaBlinder()
        params = {**DEFAULT_PARAMS, "n": 5000, "sigma": 0.5}
        rng = np.random.default_rng(42)
        data = concept.generate_data(params, rng)
        result = concept.estimate(data, params)

        assert abs(result["beta_a"][0] - params["intercept_a"]) < 0.2
        assert abs(result["beta_a"][1] - params["beta_edu_a"]) < 0.1
        assert abs(result["beta_b"][0] - params["intercept_b"]) < 0.2
        assert abs(result["beta_b"][1] - params["beta_edu_b"]) < 0.1
