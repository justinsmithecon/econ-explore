"""Base classes for all interactive concept apps."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Union

import numpy as np


# ---------------------------------------------------------------------------
# Parameter spec types
# ---------------------------------------------------------------------------

@dataclass
class SliderSpec:
    """Declarative specification for a numeric slider parameter."""
    key: str
    label: str
    min_value: float
    max_value: float
    default: float
    step: float
    help_text: str = ""


@dataclass
class SelectSpec:
    """Declarative specification for a dropdown parameter."""
    key: str
    label: str
    options: list[str]
    default: str = ""
    help_text: str = ""

    def __post_init__(self):
        if not self.default:
            self.default = self.options[0]


ParamSpec = Union[SliderSpec, SelectSpec]


# ---------------------------------------------------------------------------
# Tier 1: Common protocol — all apps implement this
# ---------------------------------------------------------------------------

class InteractiveConcept(ABC):
    """Base class for all interactive concept apps."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Display name shown in navigation and headers."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """One-paragraph explanation of the concept."""
        ...

    @abstractmethod
    def params(self) -> list[ParamSpec]:
        """Declarative parameter specifications for the sidebar."""
        ...

    @abstractmethod
    def render(self, params: dict[str, Any], depth: str = "undergraduate") -> None:
        """Render the full visualization given current parameter values.

        Args:
            params: Current parameter values keyed by ParamSpec.key.
            depth: One of "undergraduate" or "graduate".
        """
        ...

    def educational_sections(self, depth: str = "undergraduate") -> list[tuple[str, str]]:
        """Optional learn-more content as (title, markdown_body) pairs."""
        return []


# ---------------------------------------------------------------------------
# Tier 2: Category-specific bases
# ---------------------------------------------------------------------------

class HypothesisTestConcept(InteractiveConcept):
    """Base for power analysis, p-value, and CI coverage apps."""

    @abstractmethod
    def analytic_result(self, params: dict[str, Any]) -> float:
        """Compute analytic power (or coverage, etc.)."""
        ...

    @abstractmethod
    def simulate_result(
        self, params: dict[str, Any], n_simulations: int, seed: int | None
    ) -> dict[str, Any]:
        """Run Monte Carlo simulation. Must return dict with at least 'power' key."""
        ...

    @abstractmethod
    def null_distribution(self, params: dict[str, Any]) -> dict[str, Any]:
        """Return info about the null distribution for plotting."""
        ...

    @abstractmethod
    def alt_distribution(self, params: dict[str, Any]) -> dict[str, Any]:
        """Return info about the alternative distribution for plotting."""
        ...


class SamplingConcept(InteractiveConcept):
    """Base for CLT, bootstrap, LLN — apps about sampling distributions."""

    @abstractmethod
    def population_distribution(self, params: dict[str, Any]) -> dict[str, Any]:
        """Return info about the population distribution."""
        ...

    @abstractmethod
    def draw_samples(
        self, params: dict[str, Any], rng: np.random.Generator
    ) -> dict[str, Any]:
        """Draw samples and compute statistics."""
        ...


class EquilibriumConcept(InteractiveConcept):
    """Base for supply/demand, IS-LM, game theory — equilibrium models."""

    @abstractmethod
    def compute_curves(self, params: dict[str, Any]) -> dict[str, Any]:
        """Compute the curves (supply/demand, IS/LM, etc.)."""
        ...

    @abstractmethod
    def compute_equilibrium(self, params: dict[str, Any]) -> dict[str, Any]:
        """Compute the equilibrium point(s)."""
        ...


class EstimationConcept(InteractiveConcept):
    """Base for OLS, IV, MLE — apps about estimation methods."""

    @abstractmethod
    def generate_data(
        self, params: dict[str, Any], rng: np.random.Generator
    ) -> dict[str, Any]:
        """Generate synthetic data for the estimation exercise."""
        ...

    @abstractmethod
    def estimate(self, data: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        """Run the estimation procedure on the data."""
        ...
