import pandas as pd

from frmtpl_scaling.config import EXPOSURE_COL, SAMPLE_COL, TARGET_COL
from frmtpl_scaling.preprocessing import base_rate, fit_preprocessor, make_keras_data


def _sample_frame():
    return pd.DataFrame(
        {
            "Area": ["A", "B", "A", "C"],
            "VehAge": [1, 3, 5, 7],
            "Density": [10, 20, 30, 40],
            TARGET_COL: [0, 1, 0, 2],
            EXPOSURE_COL: [1.0, 1.0, 0.5, 0.5],
            SAMPLE_COL: [0.1, 0.2, 0.3, 0.4],
            "set": ["train", "train", "train", "train"],
        }
    )


def test_preprocessor_encodes_all_features_for_keras():
    preprocessor = fit_preprocessor(_sample_frame(), n_bins=2)
    encoded = preprocessor.transform(_sample_frame())
    x, y = make_keras_data(encoded, preprocessor.feature_names)

    assert "Area" in x
    assert "Exposure" in x
    assert y.shape == (4, 1)
    assert encoded["Area"].min() >= 1
    assert encoded["VehAge"].max() <= preprocessor.cardinalities["VehAge"] - 1


def test_base_rate_uses_exposure_offset():
    preprocessor = fit_preprocessor(_sample_frame(), n_bins=2)
    encoded = preprocessor.transform(_sample_frame())
    assert base_rate(encoded) == 1.0
