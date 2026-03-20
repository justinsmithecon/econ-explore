"""Tests for the Human Capital & Returns to Schooling app."""

import numpy as np

from apps.human_capital.app import HumanCapital, _compute_irr


DEFAULT = {
    "w_0": 35000, "w_1": 50000, "k": 4, "tuition": 15000,
    "g": 2.0, "r": 5.0, "T": 40,
}


class TestHumanCapital:
    def test_pv_work_now_positive(self):
        c = HumanCapital()
        result = c.compute(DEFAULT)
        assert result["pv_a"] > 0

    def test_total_tuition(self):
        c = HumanCapital()
        result = c.compute(DEFAULT)
        assert result["total_tuition"] == DEFAULT["tuition"] * DEFAULT["k"]

    def test_total_cost_includes_opportunity(self):
        """Total cost = tuition + foregone earnings."""
        c = HumanCapital()
        result = c.compute(DEFAULT)
        assert result["total_cost"] == result["total_tuition"] + result["total_opportunity"]
        assert result["total_opportunity"] > 0

    def test_no_tuition_reduces_cost(self):
        c = HumanCapital()
        r1 = c.compute(DEFAULT)
        r2 = c.compute({**DEFAULT, "tuition": 0})
        assert r2["total_cost"] < r1["total_cost"]
        assert r2["npv_education"] > r1["npv_education"]

    def test_higher_wage_premium_increases_npv(self):
        c = HumanCapital()
        r_low = c.compute({**DEFAULT, "w_1": 40000})
        r_high = c.compute({**DEFAULT, "w_1": 70000})
        assert r_high["npv_education"] > r_low["npv_education"]

    def test_higher_discount_rate_lowers_npv(self):
        c = HumanCapital()
        r_low = c.compute({**DEFAULT, "r": 3.0})
        r_high = c.compute({**DEFAULT, "r": 10.0})
        assert r_low["npv_education"] > r_high["npv_education"]

    def test_zero_discount_rate(self):
        """With r=0 and no wage growth, NPV = (T-k)*w_1 - T*w_0 - k*tuition."""
        c = HumanCapital()
        params = {**DEFAULT, "r": 0.0, "g": 0.0}
        result = c.compute(params)
        k, T = params["k"], params["T"]
        expected = (T - k) * params["w_1"] - T * params["w_0"] - k * params["tuition"]
        assert abs(result["npv_education"] - expected) < 1.0

    def test_breakeven_exists_when_npv_positive(self):
        """If education has positive NPV, there should be a crossover."""
        c = HumanCapital()
        result = c.compute({**DEFAULT, "w_1": 60000, "tuition": 5000})
        assert result["npv_education"] > 0
        assert result["breakeven"] is not None
        assert result["breakeven"] < DEFAULT["T"]

    def test_income_during_school_is_negative_tuition(self):
        c = HumanCapital()
        result = c.compute(DEFAULT)
        for t in range(DEFAULT["k"]):
            assert result["income_b"][t] == -DEFAULT["tuition"]

    def test_income_after_school(self):
        """After graduation, income_b should be w_1 * (1+g)^(t-k)."""
        c = HumanCapital()
        params = {**DEFAULT, "g": 0.0}
        result = c.compute(params)
        assert abs(result["income_b"][params["k"]] - params["w_1"]) < 0.01

    def test_irr_between_bounds(self):
        c = HumanCapital()
        result = c.compute(DEFAULT)
        if result["irr"] is not None:
            assert -0.5 < result["irr"] < 2.0

    def test_irr_makes_npv_zero(self):
        """At the IRR, NPV should be approximately zero."""
        c = HumanCapital()
        result = c.compute(DEFAULT)
        if result["irr"] is not None:
            params_at_irr = {**DEFAULT, "r": result["irr"] * 100}
            result_at_irr = c.compute(params_at_irr)
            assert abs(result_at_irr["npv_education"]) < 500  # within $500
