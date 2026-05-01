import sys
import types

import pytest

from scripts.run_experiment import _ensure_gpu_available


def test_gpu_check_is_skipped_unless_required(monkeypatch):
    monkeypatch.setitem(sys.modules, "tensorflow", None)

    _ensure_gpu_available(False)


def test_gpu_check_exits_when_tensorflow_has_no_gpu(monkeypatch):
    fake_tf = types.SimpleNamespace(
        config=types.SimpleNamespace(list_physical_devices=lambda kind: [])
    )
    monkeypatch.setitem(sys.modules, "tensorflow", fake_tf)

    with pytest.raises(SystemExit) as excinfo:
        _ensure_gpu_available(True)

    assert "No TensorFlow GPU was detected" in str(excinfo.value)


def test_gpu_check_allows_visible_tensorflow_gpu(monkeypatch, capsys):
    fake_gpu = types.SimpleNamespace(name="/physical_device:GPU:0")
    fake_tf = types.SimpleNamespace(
        config=types.SimpleNamespace(list_physical_devices=lambda kind: [fake_gpu])
    )
    monkeypatch.setitem(sys.modules, "tensorflow", fake_tf)

    _ensure_gpu_available(True)

    assert "/physical_device:GPU:0" in capsys.readouterr().out
