import pandas as pd
import pytest

from frmtpl_scaling.config import EXPOSURE_COL, SAMPLE_COL, SPLIT_COL, TARGET_COL
from frmtpl_scaling.data import load_frmtpl_csv, train_test_split_from_set


def _write_sample_csv(path, include_split=True):
    df = pd.DataFrame(
        {
            TARGET_COL: [0, 1, 0, 2],
            EXPOSURE_COL: [1.0, 0.5, 1.0, 0.25],
            "Area": ["A", "B", "A", "C"],
            SAMPLE_COL: [0.1, 0.2, 0.3, 0.4],
        }
    )
    if include_split:
        df[SPLIT_COL] = ["train", "train", "test", "test"]
    df.to_csv(path, index=False)


def test_loader_requires_supplied_learning_test_split(tmp_path):
    path = tmp_path / "sample.csv"
    _write_sample_csv(path, include_split=False)

    with pytest.raises(ValueError, match="Expected a `set` column"):
        load_frmtpl_csv(path)


def test_loader_uses_supplied_learning_test_split(tmp_path):
    path = tmp_path / "sample.csv"
    _write_sample_csv(path, include_split=True)

    df = load_frmtpl_csv(path)
    train, test = train_test_split_from_set(df)

    assert len(train) == 2
    assert len(test) == 2
