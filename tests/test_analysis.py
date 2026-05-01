import numpy as np

from frmtpl_scaling.analysis import fit_power_law
from frmtpl_scaling.config import SMOKE_THRESHOLDS


def test_power_law_fit_is_undefined_with_one_point():
    fit = fit_power_law(np.array([100.0]), np.array([0.25]))

    assert np.isnan(fit["alpha"])


def test_power_law_fit_returns_exponent_with_two_points():
    fit = fit_power_law(np.array([100.0, 200.0]), np.array([0.30, 0.25]))

    assert np.isfinite(fit["l_inf"])
    assert np.isfinite(fit["alpha"])
    assert np.isfinite(fit["a"])


def test_smoke_thresholds_support_scaling_fit():
    assert len(SMOKE_THRESHOLDS) >= 2
