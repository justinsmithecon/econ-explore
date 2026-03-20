"""Proportion z-test power scenario."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

from shared.stats_utils import power_from_normal_approx
from shared.base import SliderSpec, HypothesisTestConcept


class ProportionTest(HypothesisTestConcept):
    @property
    def name(self) -> str:
        return "Proportion (z-test)"

    @property
    def description(self) -> str:
        return (
            "Test whether a population proportion differs from a hypothesized value. "
            "For example, does a new website design change the conversion rate from "
            "the baseline of 10%?"
        )

    def params(self) -> list[SliderSpec]:
        return [
            SliderSpec("p_0", "Null proportion (p₀)", 0.01, 0.99, 0.10, 0.01,
                       "Hypothesized proportion under H₀"),
            SliderSpec("p_1", "True proportion (p₁)", 0.01, 0.99, 0.15, 0.01,
                       "The actual proportion you expect"),
            SliderSpec("n", "Sample size (n)", 10.0, 1000.0, 200.0, 10.0,
                       "Number of observations"),
            SliderSpec("alpha", "Significance level (α)", 0.01, 0.20, 0.05, 0.01,
                       "Type I error rate"),
        ]

    def render(self, params: dict[str, Any], depth: str = "undergraduate") -> None:
        pass

    def analytic_result(self, params: dict[str, Any]) -> float:
        return self.analytic_power(params)

    def simulate_result(self, params: dict[str, Any], n_simulations: int, seed: int | None) -> dict[str, Any]:
        return self.simulate_power(params, n_simulations, seed)

    def null_distribution(self, params: dict[str, Any]) -> dict[str, Any]:
        return self.distribution_info(params)

    def alt_distribution(self, params: dict[str, Any]) -> dict[str, Any]:
        return self.test_statistic_info(params)

    def sliders(self) -> list[SliderSpec]:
        return self.params()

    def analytic_power(self, params: dict[str, Any]) -> float:
        p_0, p_1 = params["p_0"], params["p_1"]
        n, alpha = int(params["n"]), params["alpha"]
        se_0 = np.sqrt(p_0 * (1 - p_0) / n)
        z_ncp = (p_1 - p_0) / se_0
        return power_from_normal_approx(z_ncp, alpha)

    def simulate_power(
        self, params: dict[str, Any], n_simulations: int = 5000, seed: int | None = None
    ) -> dict[str, Any]:
        p_0, p_1 = params["p_0"], params["p_1"]
        n, alpha = int(params["n"]), params["alpha"]
        se_0 = np.sqrt(p_0 * (1 - p_0) / n)
        rng = np.random.default_rng(seed)

        sample_stats = np.empty(n_simulations)
        test_stats = np.empty(n_simulations)
        p_values = np.empty(n_simulations)

        for i in range(n_simulations):
            data = rng.binomial(1, p_1, size=n)
            p_hat = data.mean()
            z = (p_hat - p_0) / se_0 if se_0 > 0 else 0.0
            p_val = float(2 * (1 - stats.norm.cdf(abs(z))))
            sample_stats[i] = p_hat
            test_stats[i] = z
            p_values[i] = p_val

        power = (p_values < alpha).mean()
        return {"power": float(power), "p_values": p_values,
                "sample_stats": sample_stats, "test_stats": test_stats}

    def formula_latex(self) -> str:
        return (
            r"z = \frac{\hat{p} - p_0}{\sqrt{p_0(1-p_0)/n}}"
            r"\quad\text{NCP} = \frac{p_1 - p_0}{\sqrt{p_0(1-p_0)/n}}"
        )

    def distribution_info(self, params: dict[str, Any]) -> dict[str, Any]:
        p_0, p_1 = params["p_0"], params["p_1"]
        n, alpha = int(params["n"]), params["alpha"]
        se_0 = np.sqrt(p_0 * (1 - p_0) / n)
        se_1 = np.sqrt(p_1 * (1 - p_1) / n)
        return {
            "null_dist": stats.norm(p_0, se_0),
            "true_dist": stats.norm(p_1, se_1),
            "alpha": alpha,
            "stat_name": "Sample proportion (p̂)",
            "stat_formula": r"\hat{p} \sim \mathcal{N}\!\left(p,\; \sqrt{\frac{p(1-p)}{n}}\right)",
        }

    def test_statistic_info(self, params: dict[str, Any]) -> dict[str, Any]:
        p_0, p_1 = params["p_0"], params["p_1"]
        n, alpha = int(params["n"]), params["alpha"]
        se_0 = np.sqrt(p_0 * (1 - p_0) / n)
        z_ncp = (p_1 - p_0) / se_0
        return {
            "null_dist": stats.norm(0, 1),
            "true_dist": stats.norm(z_ncp, 1),
            "alpha": alpha,
            "stat_name": "z-statistic",
            "stat_formula": r"z = \frac{\hat{p} - p_0}{\sqrt{p_0(1-p_0)/n}}"
                            r"\quad\text{Under } H_0: z \sim \mathcal{N}(0, 1)"
                            r"\quad\text{Under } H_1: z \sim \mathcal{N}(\delta, 1)",
        }

    def power_curve_data(self, params: dict[str, Any], n_range: np.ndarray | None = None) -> dict[str, Any]:
        if n_range is None:
            n_range = np.arange(10, 210, 10)
        n_key = "n_per_group" if "n_per_group" in params else "n"
        analytic_powers = []
        for n in n_range:
            p = {**params, n_key: int(n)}
            analytic_powers.append(self.analytic_power(p))
        return {"n_range": n_range, "analytic": np.array(analytic_powers)}
