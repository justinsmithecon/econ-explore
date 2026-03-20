"""Generic vectorized Monte Carlo simulation runner."""

from __future__ import annotations

from typing import Callable

import numpy as np


def run_simulation(
    data_generator: Callable[[np.random.Generator], np.ndarray],
    test_function: Callable[[np.ndarray], float],
    n_simulations: int = 5000,
    alpha: float = 0.05,
    seed: int | None = None,
) -> dict:
    """
    Run a Monte Carlo power simulation.

    Returns dict with keys: power, ci_lower, ci_upper, p_values
    """
    rng = np.random.default_rng(seed)
    p_values = np.array([test_function(data_generator(rng)) for _ in range(n_simulations)])
    rejections = p_values < alpha
    power = rejections.mean()

    # Wilson score interval for proportion
    z = 1.96
    n = n_simulations
    denom = 1 + z**2 / n
    center = (power + z**2 / (2 * n)) / denom
    half_width = z * np.sqrt((power * (1 - power) + z**2 / (4 * n)) / n) / denom
    ci_lower = max(0, center - half_width)
    ci_upper = min(1, center + half_width)

    return {
        "power": float(power),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "p_values": p_values,
    }
