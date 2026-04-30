import numpy as np

from frmtpl_scaling.losses import poisson_deviance


def test_poisson_deviance_is_zero_for_perfect_positive_predictions():
    y = np.array([0.0, 1.0, 2.0])
    assert poisson_deviance(y, y + 1e-7) < 1e-5


def test_poisson_deviance_penalizes_bad_predictions():
    y = np.array([0.0, 1.0, 2.0])
    good = poisson_deviance(y, np.array([0.1, 1.0, 2.0]))
    bad = poisson_deviance(y, np.array([2.0, 0.1, 0.1]))
    assert bad > good
