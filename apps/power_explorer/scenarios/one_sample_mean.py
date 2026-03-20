"""One-sample t-test power scenario."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

from shared.stats_utils import power_from_noncentral_t
from shared.base import SliderSpec, HypothesisTestConcept


class OneSampleMean(HypothesisTestConcept):
    @property
    def name(self) -> str:
        return "One-Sample Mean (t-test)"

    @property
    def description(self) -> str:
        return (
            "Test whether a population mean differs from a hypothesized value. "
            "For example, does a new teaching method change average test scores "
            "from the historical average of 70?"
        )

    def params(self) -> list[SliderSpec]:
        return [
            SliderSpec("mu_0", "Null hypothesis mean (μ₀)", 0.0, 100.0, 50.0, 1.0,
                       "The hypothesized population mean under H₀"),
            SliderSpec("mu_1", "True mean (μ₁)", 0.0, 100.0, 55.0, 1.0,
                       "The actual population mean you expect"),
            SliderSpec("sigma", "Standard deviation (σ)", 1.0, 50.0, 10.0, 0.5,
                       "Population standard deviation"),
            SliderSpec("n", "Sample size (n)", 5.0, 300.0, 30.0, 5.0,
                       "Number of observations"),
            SliderSpec("alpha", "Significance level (α)", 0.01, 0.20, 0.05, 0.01,
                       "Type I error rate"),
        ]

    def render(self, params: dict[str, Any], depth: str = "undergraduate") -> None:
        pass  # Rendering handled by the power explorer app orchestrator

    def analytic_result(self, params: dict[str, Any]) -> float:
        return self.analytic_power(params)

    def simulate_result(
        self, params: dict[str, Any], n_simulations: int, seed: int | None
    ) -> dict[str, Any]:
        return self.simulate_power(params, n_simulations, seed)

    def null_distribution(self, params: dict[str, Any]) -> dict[str, Any]:
        return self.distribution_info(params)

    def alt_distribution(self, params: dict[str, Any]) -> dict[str, Any]:
        return self.test_statistic_info(params)

    # --- Power scenario methods (used by the app orchestrator) ---

    def sliders(self) -> list[SliderSpec]:
        return self.params()

    def analytic_power(self, params: dict[str, Any]) -> float:
        mu_0, mu_1, sigma = params["mu_0"], params["mu_1"], params["sigma"]
        n, alpha = int(params["n"]), params["alpha"]
        ncp = (mu_1 - mu_0) / (sigma / np.sqrt(n))
        df = n - 1
        return power_from_noncentral_t(ncp, df, alpha)

    def simulate_power(
        self, params: dict[str, Any], n_simulations: int = 5000, seed: int | None = None
    ) -> dict[str, Any]:
        mu_0, mu_1, sigma = params["mu_0"], params["mu_1"], params["sigma"]
        n, alpha = int(params["n"]), params["alpha"]
        rng = np.random.default_rng(seed)

        sample_stats = np.empty(n_simulations)
        test_stats = np.empty(n_simulations)
        p_values = np.empty(n_simulations)

        for i in range(n_simulations):
            data = rng.normal(mu_1, sigma, size=n)
            x_bar = data.mean()
            s = data.std(ddof=1)
            t_stat = (x_bar - mu_0) / (s / np.sqrt(n))
            _, p = stats.ttest_1samp(data, mu_0)
            sample_stats[i] = x_bar
            test_stats[i] = t_stat
            p_values[i] = p

        power = (p_values < alpha).mean()
        return {"power": float(power), "p_values": p_values,
                "sample_stats": sample_stats, "test_stats": test_stats}

    def formula_latex(self) -> str:
        return (
            r"\text{Non-centrality parameter: } \delta = \frac{\mu_1 - \mu_0}{\sigma / \sqrt{n}}"
            r"\quad\text{df} = n - 1"
        )

    def distribution_info(self, params: dict[str, Any]) -> dict[str, Any]:
        mu_0, mu_1, sigma = params["mu_0"], params["mu_1"], params["sigma"]
        n, alpha = int(params["n"]), params["alpha"]
        se = sigma / np.sqrt(n)
        return {
            "null_dist": stats.norm(mu_0, se),
            "true_dist": stats.norm(mu_1, se),
            "alpha": alpha,
            "stat_name": "Sample mean (x̄)",
            "stat_formula": r"\bar{x} \sim \mathcal{N}\!\left(\mu,\; \frac{\sigma}{\sqrt{n}}\right)",
        }

    def test_statistic_info(self, params: dict[str, Any]) -> dict[str, Any]:
        mu_0, mu_1, sigma = params["mu_0"], params["mu_1"], params["sigma"]
        n, alpha = int(params["n"]), params["alpha"]
        ncp = (mu_1 - mu_0) / (sigma / np.sqrt(n))
        df = n - 1
        return {
            "null_dist": stats.t(df),
            "true_dist": stats.nct(df, ncp),
            "alpha": alpha,
            "stat_name": "t-statistic",
            "stat_formula": r"t = \frac{\bar{x} - \mu_0}{s / \sqrt{n}}"
                            r"\quad\text{Under } H_0: t \sim t_{n-1}"
                            r"\quad\text{Under } H_1: t \sim t_{n-1}(\delta)",
        }

    def power_curve_data(
        self,
        params: dict[str, Any],
        n_range: np.ndarray | None = None,
    ) -> dict[str, Any]:
        if n_range is None:
            n_range = np.arange(10, 210, 10)
        n_key = "n_per_group" if "n_per_group" in params else "n"
        analytic_powers = []
        for n in n_range:
            p = {**params, n_key: int(n)}
            analytic_powers.append(self.analytic_power(p))
        return {"n_range": n_range, "analytic": np.array(analytic_powers)}
