import numpy as np
import pytest

from src.dataset import BG_PALETTE, N_CLASSES, luminance, make_dataset


def test_shapes_labels_and_range():
    d = make_dataset(120, seed=1)
    assert d.x.shape == (120, 3, 32, 32) and d.x.dtype == np.float32
    assert d.x.min() >= 0 and d.x.max() <= 1
    assert set(np.unique(d.y)) == set(range(N_CLASSES))
    assert set(np.unique(d.bg)) <= set(range(N_CLASSES))
    assert np.bincount(d.y).tolist() == [40, 40, 40]  # balanced


def test_correlation_strength_is_respected():
    d = make_dataset(3000, seed=2, bg_mode="correlated", correlation=0.9)
    assert abs((d.bg == d.y).mean() - 0.9) < 0.02
    assert (make_dataset(300, 2, "reversed").bg == make_dataset(300, 2, "reversed").y).sum() == 0


def test_random_mode_is_independent_of_label():
    d = make_dataset(3000, seed=3, bg_mode="random")
    assert abs((d.bg == d.y).mean() - 1 / 3) < 0.04


def test_same_objects_across_modes():
    a, b = make_dataset(90, 4, "correlated"), make_dataset(90, 4, "reversed")
    assert np.array_equal(a.y, b.y)
    assert not np.array_equal(a.x, b.x)  # backgrounds differ


def test_reproducible_for_fixed_seed_and_differs_across_seeds():
    assert np.array_equal(make_dataset(60, 5).x, make_dataset(60, 5).x)
    assert not np.array_equal(make_dataset(60, 5).x, make_dataset(60, 6).x)


def test_background_hue_is_the_cue_not_brightness():
    lum = luminance(BG_PALETTE)
    assert lum.max() - lum.min() < 0.05


def test_invalid_arguments():
    with pytest.raises(ValueError):
        make_dataset(10, 0, bg_mode="nope")
    with pytest.raises(ValueError):
        make_dataset(10, 0, correlation=1.5)
    with pytest.raises(ValueError):
        make_dataset(0, 0)
