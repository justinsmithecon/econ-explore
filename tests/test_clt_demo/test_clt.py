"""Tests for the CLT demo concept."""

import numpy as np

from apps.clt_demo.app import CLTDemo, POPULATIONS


class TestCLTDemo:
    def test_populations_have_required_keys(self):
        for name, pop in POPULATIONS.items():
            assert "rvs" in pop, f"{name} missing rvs"
            assert "mean" in pop, f"{name} missing mean"
            assert "std" in pop, f"{name} missing std"

    def test_draw_samples_returns_correct_count(self):
        concept = CLTDemo()
        rng = np.random.default_rng(42)
        result = concept.draw_samples(
            {"population": "Uniform (0, 1)", "n": 10, "n_samples": 500}, rng
        )
        assert len(result["means"]) == 500

    def test_sample_mean_converges_to_population_mean(self):
        concept = CLTDemo()
        rng = np.random.default_rng(42)
        for pop_name, pop in POPULATIONS.items():
            result = concept.draw_samples(
                {"population": pop_name, "n": 100, "n_samples": 5000}, rng
            )
            assert abs(result["means"].mean() - pop["mean"]) < 0.1, (
                f"{pop_name}: mean of sample means too far from population mean"
            )

    def test_se_decreases_with_n(self):
        concept = CLTDemo()
        rng = np.random.default_rng(42)
        r1 = concept.draw_samples(
            {"population": "Exponential (λ=1)", "n": 10, "n_samples": 5000}, rng
        )
        rng = np.random.default_rng(42)
        r2 = concept.draw_samples(
            {"population": "Exponential (λ=1)", "n": 100, "n_samples": 5000}, rng
        )
        assert r2["means"].std() < r1["means"].std()
