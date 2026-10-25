import json

import numpy as np
import pytest
import torch

import src.experiment as ex
from src.experiment import (ExperimentConfig, accuracy, color_augment, confusion_matrix, per_class_accuracy,
                            robustness_gap, run_experiment, wilson_interval)

TINY = dict(n_train=150, n_val=60, n_test=90, epochs=2, batch_size=32)


def test_accuracy_gap_confusion_math():
    pred = np.array([0, 1, 2, 2, 1, 0])
    true = np.array([0, 1, 2, 0, 1, 1])
    assert accuracy(pred, true) == pytest.approx(4 / 6)
    assert robustness_gap(0.95, 0.40) == pytest.approx(0.55)
    cm = confusion_matrix(pred, true)  # rows = true class, cols = predicted
    assert cm.tolist() == [[1, 0, 1], [1, 2, 0], [0, 0, 1]]
    assert per_class_accuracy(cm) == pytest.approx([0.5, 2 / 3, 1.0])
    with pytest.raises(ValueError):
        accuracy(np.array([]), np.array([]))


def test_wilson_interval_contains_estimate():
    lo, hi = wilson_interval(80, 100)
    assert lo < 0.8 < hi and 0 <= lo and hi <= 1


def test_config_validation():
    for bad in (dict(correlation=1.2), dict(aug_hue_p=1.5), dict(aug_perm_p=-0.1), dict(epochs=0), dict(n_train=5), dict(conditions=[]), dict(conditions=["X"]), dict(lr=-1)):
        with pytest.raises(ValueError):
            ExperimentConfig(**bad).validate()
    ExperimentConfig().validate()


def test_color_augment_preserves_shape_and_range():
    x = torch.rand(8, 3, 32, 32)
    out = color_augment(x, ExperimentConfig(), torch.Generator().manual_seed(0))
    assert out.shape == x.shape and out.min() >= 0 and out.max() <= 1
    assert not torch.equal(out, x)


def test_training_never_builds_test_sets(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("test data requested during training")

    monkeypatch.setattr(ex, "build_eval_sets", boom)
    model, hist = ex.train_condition("A_baseline", ExperimentConfig(**TINY), seed=0)
    assert len(hist) == 2


def test_full_run_saves_and_loads(tmp_path):
    res = run_experiment(ExperimentConfig(seed=0, **TINY), results_root=tmp_path)
    assert res["status"] == "complete" and len(res["runs"]) == 3
    assert set(res["summary"]) == set(ex.DEFAULT_CONDITIONS)
    for r in res["runs"]:
        m = r["metrics"]
        assert r["robustness_gap"] == pytest.approx(m["id"]["accuracy"] - m["shifted"]["accuracy"])
        assert np.array(m["id"]["confusion"]).sum() == 90
        assert (tmp_path / res["run_id"] / r["model_path"]).exists()
    assert ex.list_runs(tmp_path) == [res["run_id"]]
    loaded = ex.load_results(res["run_id"], tmp_path)
    assert loaded["config"] == res["config"]
    model = ex.load_model(res["run_id"], res["runs"][0]["model_path"], tmp_path)
    ds = ex.build_eval_sets(ex.config_from_dict(loaded["config"]), 0)["id"]
    assert accuracy(ex.predict(model, ds.x), ds.y) == pytest.approx(res["runs"][0]["metrics"]["id"]["accuracy"])


def test_reproducible_for_fixed_seed(tmp_path):
    cfg = ExperimentConfig(seed=3, conditions=["A_baseline", "C_color_aug"], **TINY)
    a = run_experiment(cfg, results_root=tmp_path / "a", save_models=False)
    b = run_experiment(cfg, results_root=tmp_path / "b", save_models=False)
    for ra, rb in zip(a["runs"], b["runs"]):
        for k in ex.EVAL_SETS:
            assert ra["metrics"][k]["accuracy"] == pytest.approx(rb["metrics"][k]["accuracy"], abs=0.02)
        assert ra["history"][-1]["train_loss"] == pytest.approx(rb["history"][-1]["train_loss"], rel=1e-3)


def test_failure_is_reported_not_hidden(tmp_path, monkeypatch):
    def broken(*a, **k):
        raise RuntimeError("simulated crash")

    monkeypatch.setattr(ex, "train_condition", broken)
    res = run_experiment(ExperimentConfig(**TINY), results_root=tmp_path)
    assert res["status"] == "failed" and "simulated crash" in res["error"] and res["runs"] == []
    saved = json.loads((tmp_path / res["run_id"] / "results.json").read_text())
    assert saved["status"] == "failed"


def test_invalid_config_raises_before_running(tmp_path):
    with pytest.raises(ValueError):
        run_experiment(ExperimentConfig(correlation=2.0), results_root=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_hue_rotation_matrices_identity_and_luminance():
    theta = torch.tensor([0.0, 2 * np.pi, 1.0, 2.5])
    m = ex.hue_rotation_matrices(theta)
    assert m.shape == (4, 3, 3)
    assert torch.allclose(m[0], torch.eye(3), atol=1e-4) and torch.allclose(m[1], torch.eye(3), atol=1e-4)
    lum = torch.tensor([0.299, 0.587, 0.114])
    v = torch.tensor([0.5, 0.45, 0.55])
    for k in range(4):  # rotation must not change luminance
        assert float(lum @ (m[k] @ v)) == pytest.approx(float(lum @ v), abs=1e-3)


def test_hue_augment_shape_range_and_grey_invariance():
    cfg = ExperimentConfig()
    x = torch.rand(8, 3, 32, 32) * 0.2 + 0.4
    out = ex.hue_augment(x, cfg, torch.Generator().manual_seed(0))
    assert out.shape == x.shape and out.min() >= 0 and out.max() <= 1
    assert not torch.equal(out, x)
    grey = torch.full((4, 3, 8, 8), 0.7)  # achromatic images are unchanged by hue rotation
    assert torch.allclose(ex.hue_augment(grey, cfg, torch.Generator().manual_seed(1)), grey, atol=2e-3)
    off = ExperimentConfig(aug_hue_p=0.0)  # p = 0 -> identity
    assert torch.allclose(ex.hue_augment(x, off, torch.Generator().manual_seed(2)), x, atol=1e-4)


def test_condition_d_trains_and_is_opt_in(tmp_path):
    assert "D_hue_aug" in ex.CONDITIONS and "D_hue_aug" not in ex.DEFAULT_CONDITIONS
    assert "D_hue_aug" not in ExperimentConfig().conditions
    res = run_experiment(ExperimentConfig(conditions=["D_hue_aug"], **TINY), results_root=tmp_path, save_models=False)
    assert res["status"] == "complete" and list(res["summary"]) == ["D_hue_aug"]
