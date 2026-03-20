from apps.power_explorer.scenarios.one_sample_mean import OneSampleMean
from apps.power_explorer.scenarios.two_sample_mean import TwoSampleMean
from apps.power_explorer.scenarios.proportion import ProportionTest
from apps.power_explorer.scenarios.regression_slope import RegressionSlope

ALL_SCENARIOS = [
    OneSampleMean(),
    TwoSampleMean(),
    ProportionTest(),
    RegressionSlope(),
]
