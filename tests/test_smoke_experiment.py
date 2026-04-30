import numpy as np
import pandas as pd

from frmtpl_scaling.config import EXPOSURE_COL, SAMPLE_COL, SPLIT_COL, TARGET_COL
from frmtpl_scaling.train import run_experiment


def test_tiny_end_to_end_experiment_writes_outputs(tmp_path):
    rng = np.random.default_rng(123)
    rows = []
    for i in range(36):
        is_train = i < 28
        exposure = 0.25 + 0.75 * float(rng.uniform())
        driv_age = int(rng.choice([22, 35, 48, 61]))
        bonus_malus = int(rng.choice([50, 75, 100]))
        expected = exposure * (0.04 + 0.001 * (bonus_malus - 50) + (0.04 if driv_age < 25 else 0.0))
        rows.append(
            {
                "IDpol": i + 1,
                TARGET_COL: int(rng.poisson(expected)),
                EXPOSURE_COL: exposure,
                "Area": str(rng.choice(["A", "B", "C"])),
                "VehAge": int(rng.integers(0, 12)),
                "DrivAge": driv_age,
                "BonusMalus": bonus_malus,
                "Density": float(rng.uniform(10, 2000)),
                SPLIT_COL: "train" if is_train else "test",
                SAMPLE_COL: float(rng.uniform()) if is_train else 1.0,
            }
        )

    data_path = tmp_path / "tiny_frmtpl.csv"
    results_dir = tmp_path / "results"
    pd.DataFrame(rows).to_csv(data_path, index=False)

    run_df, ensemble_df, scaling_df = run_experiment(
        data_path=data_path,
        results_dir=results_dir,
        models="glm",
        thresholds=[1.0],
        reps=1,
        epochs=1,
        seeds=[1],
    )

    assert len(run_df) == 1
    assert len(ensemble_df) == 1
    assert len(scaling_df) == 1
    assert np.isfinite(ensemble_df["test_poisson_deviance"].iloc[0])
    assert (results_dir / "run_scores.csv").exists()
    assert (results_dir / "ensemble_scores.csv").exists()
    assert (results_dir / "scaling_fits.csv").exists()
