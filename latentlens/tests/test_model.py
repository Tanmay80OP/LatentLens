import pytest
import torch

from src.model import SmallCNN, count_parameters


def test_forward_and_embedding_dimensions():
    m = SmallCNN(3)
    x = torch.rand(5, 3, 32, 32)
    assert m(x).shape == (5, 3)
    assert m.embed(x).shape == (5, 64)
    assert count_parameters(m) < 30_000


def test_feature_maps():
    m = SmallCNN(3)
    x = torch.rand(2, 3, 32, 32)
    assert m.feature_maps(x, "conv1").shape == (2, 16, 32, 32)
    assert m.feature_maps(x, "conv2").shape == (2, 32, 16, 16)
    assert m.feature_maps(x, "conv3").shape == (2, 64, 8, 8)
    with pytest.raises(ValueError):
        m.feature_maps(x, "conv9")
