"""Two-sample t-test power scenario."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

from shared.stats_utils import power_from_noncentral_t
from shared.base import SliderSpec, HypothesisTestConcept


class TwoSampleMean(HypothesisTestConcept):
    @property
    def name(self) -> str:
        return "Two-Sample Mean (t-test)"

    @property
    def description(self) -> str:
        return (
            "Test whether two population means differ. "
            "For example, does a treatment group have a different outcome "
            "than a control group? Assumes equal variances."
        )

    def params(self) -> list[SliderSpec]:
        return [
            SliderSpec("mu_1", "Group 1 mean (μ₁)", 0.0, 100.0, 50.0, 1.0,
                       "Mean of the first (control) group"),
            SliderSpec("mu_2", "Group 2 mean (μ₂)", 0.0, 100.0, 55.0, 1.0,
                       "Mean of the second (treatment) group"),
            SliderSpec("sigma", "Common std dev (σ)", 1.0, 50.0, 10.0, 0.5,
                       "Common standard deviation (equal variance assumed)"),
            SliderSpec("n_per_group", "Sample size per group", 5.0, 300.0, 30.0, 5.0,
                       "Number of observations in each group"),
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
        mu_1, mu_2, sigma = params["mu_1"], params["mu_2"], params["sigma"]
        n, alpha = int(params["n_per_group"]), params["alpha"]
        ncp = (mu_1 - mu_2) / (sigma * np.sqrt(2 / n))
        df = 2 * n - 2
        return power_from_noncentral_t(ncp, df, alpha)

    def simulate_power(
        self, params: dict[str, Any], n_simulations: int = 5000, seed: int | None = None
    ) -> dict[str, Any]:
        mu_1, mu_2, sigma = params["mu_1"], params["mu_2"], params["sigma"]
        n, alpha = int(params["n_per_group"]), params["alpha"]
        rng = np.random.default_rng(seed)

        sample_stats = np.empty(n_simulations)
        test_stats = np.empty(n_simulations)
        p_values = np.empty(n_simulations)

        for i in range(n_simulations):
            g1 = rng.normal(mu_1, sigma, size=n)
            g2 = rng.normal(mu_2, sigma, size=n)
            diff = g1.mean() - g2.mean()
            t_result = stats.ttest_ind(g1, g2)
            sample_stats[i] = diff
            test_stats[i] = t_result.statistic
            p_values[i] = t_result.pvalue

        power = (p_values < alpha).mean()
        return {"power": float(power), "p_values": p_values,
                "sample_stats": sample_stats, "test_stats": test_stats}

    def formula_latex(self) -> str:
        return (
            r"\text{Non-centrality parameter: } \delta = \frac{\mu_1 - \mu_2}{\sigma \sqrt{2/n}}"
            r"\quad\text{df} = 2n - 2"
        )

    def distribution_info(self, params: dict[str, Any]) -> dict[str, Any]:
        mu_1, mu_2, sigma = params["mu_1"], params["mu_2"], params["sigma"]
        n, alpha = int(params["n_per_group"]), params["alpha"]
        se = sigma * np.sqrt(2 / n)
        return {
            "null_dist": stats.norm(0, se),
            "true_dist": stats.norm(mu_1 - mu_2, se),
            "alpha": alpha,
            "stat_name": "Difference in means (x̄₁ − x̄₂)",
            "stat_formula": r"\bar{x}_1 - \bar{x}_2 \sim \mathcal{N}\!\left(\mu_1 - \mu_2,\; \sigma\sqrt{\frac{2}{n}}\right)",
        }

    def test_statistic_info(self, params: dict[str, Any]) -> dict[str, Any]:
        mu_1, mu_2, sigma = params["mu_1"], params["mu_2"], params["sigma"]
        n, alpha = int(params["n_per_group"]), params["alpha"]
        ncp = (mu_1 - mu_2) / (sigma * np.sqrt(2 / n))
        df = 2 * n - 2
        return {
            "null_dist": stats.t(df),
            "true_dist": stats.nct(df, ncp),
            "alpha": alpha,
            "stat_name": "t-statistic",
            "stat_formula": r"t = \frac{\bar{x}_1 - \bar{x}_2}{s_p \sqrt{2/n}}"
                            r"\quad\text{Under } H_0: t \sim t_{2n-2}"
                            r"\quad\text{Under } H_1: t \sim t_{2n-2}(\delta)",
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
