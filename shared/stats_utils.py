"""Shared analytic utilities for power computation."""

import numpy as np
from scipy import stats


def power_from_noncentral_t(ncp: float, df: float, alpha: float = 0.05) -> float:
    """Compute power using the non-central t distribution (two-sided test)."""
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    power = 1 - stats.nct.cdf(t_crit, df, ncp) + stats.nct.cdf(-t_crit, df, ncp)
    return float(np.clip(power, 0, 1))


def power_from_normal_approx(z_ncp: float, alpha: float = 0.05) -> float:
    """Compute power using normal approximation (two-sided z-test)."""
    z_crit = stats.norm.ppf(1 - alpha / 2)
    power = 1 - stats.norm.cdf(z_crit - z_ncp) + stats.norm.cdf(-z_crit - z_ncp)
    return float(np.clip(power, 0, 1))
