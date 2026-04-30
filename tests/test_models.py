import numpy as np

from frmtpl_scaling.config import get_default_model_configs
from frmtpl_scaling.models import build_model, set_global_seed


def _toy_inputs():
    feature_names = ["Area", "VehAge"]
    cardinalities = {"Area": 4, "VehAge": 5}
    x = {
        "Area": np.array([[1], [2], [1]], dtype="int32"),
        "VehAge": np.array([[1], [2], [3]], dtype="int32"),
        "Exposure": np.array([[1.0], [0.5], [1.0]], dtype="float32"),
    }
    return feature_names, cardinalities, x


def test_all_default_models_build_and_predict_one_batch():
    feature_names, cardinalities, x = _toy_inputs()
    for name, config in get_default_model_configs().items():
        set_global_seed(123)
        model = build_model(name, config, feature_names, cardinalities, base_rate=0.1)
        pred = model.predict(x, verbose=0)
        assert pred.shape == (3, 1)
        assert np.all(np.isfinite(pred))
        assert np.all(pred > 0)


def test_tabm_mini_trains_one_batch():
    feature_names, cardinalities, x = _toy_inputs()
    config = get_default_model_configs()["tabm_mini_small"]
    y = np.array([[0.0], [1.0], [0.0]], dtype="float32")

    set_global_seed(123)
    model = build_model("tabm_mini_small", config, feature_names, cardinalities, base_rate=0.1)
    history = model.fit(x, y, epochs=1, batch_size=3, verbose=0)

    assert np.isfinite(history.history["loss"][-1])
