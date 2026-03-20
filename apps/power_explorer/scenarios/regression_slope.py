"""Regression slope t-test power scenario."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

from shared.stats_utils import power_from_noncentral_t
from shared.base import SliderSpec, HypothesisTestConcept


class RegressionSlope(HypothesisTestConcept):
    @property
    def name(self) -> str:
        return "Regression Slope (t-test)"

    @property
    def description(self) -> str:
        return (
            "Test whether a regression slope (β) differs from zero. "
            "For example, does study time (hours) predict exam score? "
            "The power depends on the true slope, the variability of x, "
            "and the noise level."
        )

    def params(self) -> list[SliderSpec]:
        return [
            SliderSpec("beta", "True slope (β)", 0.0, 5.0, 0.5, 0.1,
                       "The true regression coefficient"),
            SliderSpec("sigma_x", "Std dev of x (σ_x)", 0.5, 10.0, 2.0, 0.5,
                       "Standard deviation of the predictor variable"),
            SliderSpec("sigma_eps", "Noise std dev (σ_ε)", 0.5, 20.0, 5.0, 0.5,
                       "Standard deviation of residual errors"),
            SliderSpec("n", "Sample size (n)", 10.0, 300.0, 50.0, 5.0,
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
        beta = params["beta"]
        sigma_x, sigma_eps = params["sigma_x"], params["sigma_eps"]
        n, alpha = int(params["n"]), params["alpha"]
        ncp = beta * sigma_x * np.sqrt(n) / sigma_eps
        df = n - 2
        return power_from_noncentral_t(ncp, df, alpha)

    def simulate_power(
        self, params: dict[str, Any], n_simulations: int = 5000, seed: int | None = None
    ) -> dict[str, Any]:
        beta = params["beta"]
        sigma_x, sigma_eps = params["sigma_x"], params["sigma_eps"]
        n, alpha = int(params["n"]), params["alpha"]
        rng = np.random.default_rng(seed)

        sample_stats = np.empty(n_simulations)
        test_stats = np.empty(n_simulations)
        p_values = np.empty(n_simulations)

        for i in range(n_simulations):
            x = rng.normal(0, sigma_x, size=n)
            eps = rng.normal(0, sigma_eps, size=n)
            y = beta * x + eps
            result = stats.linregress(x, y)
            sample_stats[i] = result.slope
            test_stats[i] = result.slope / result.stderr
            p_values[i] = result.pvalue

        power = (p_values < alpha).mean()
        return {"power": float(power), "p_values": p_values,
                "sample_stats": sample_stats, "test_stats": test_stats}

    def formula_latex(self) -> str:
        return (
            r"\text{NCP} = \frac{\beta \cdot \sigma_x \cdot \sqrt{n}}{\sigma_\varepsilon}"
            r"\quad\text{df} = n - 2"
        )

    def distribution_info(self, params: dict[str, Any]) -> dict[str, Any]:
        beta = params["beta"]
        sigma_x, sigma_eps = params["sigma_x"], params["sigma_eps"]
        n, alpha = int(params["n"]), params["alpha"]
        se_beta = sigma_eps / (sigma_x * np.sqrt(n))
        return {
            "null_dist": stats.norm(0, se_beta),
            "true_dist": stats.norm(beta, se_beta),
            "alpha": alpha,
            "stat_name": "Estimated slope (β̂)",
            "stat_formula": r"\hat{\beta} \sim \mathcal{N}\!\left(\beta,\; \frac{\sigma_\varepsilon}{\sigma_x \sqrt{n}}\right)",
        }

    def test_statistic_info(self, params: dict[str, Any]) -> dict[str, Any]:
        beta = params["beta"]
        sigma_x, sigma_eps = params["sigma_x"], params["sigma_eps"]
        n, alpha = int(params["n"]), params["alpha"]
        ncp = beta * sigma_x * np.sqrt(n) / sigma_eps
        df = n - 2
        return {
            "null_dist": stats.t(df),
            "true_dist": stats.nct(df, ncp),
            "alpha": alpha,
            "stat_name": "t-statistic",
            "stat_formula": r"t = \frac{\hat{\beta}}{\text{SE}(\hat{\beta})}"
                            r"\quad\text{Under } H_0: t \sim t_{n-2}"
                            r"\quad\text{Under } H_1: t \sim t_{n-2}(\delta)",
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
