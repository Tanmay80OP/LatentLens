"""Small CNN (~25k parameters) used for every experimental condition."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

LAYER_NAMES = ("conv1", "conv2", "conv3")


class SmallCNN(nn.Module):
    def __init__(self, n_classes: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc = nn.Linear(64, n_classes)
        self.n_classes = n_classes

    def _trunk(self, x: torch.Tensor, upto: str | None = None):
        """Run conv blocks. If ``upto`` is a layer name, return that layer's post-ReLU maps."""
        x = F.relu(self.conv1(x))
        if upto == "conv1":
            return x
        x = F.relu(self.conv2(F.max_pool2d(x, 2)))
        if upto == "conv2":
            return x
        x = F.relu(self.conv3(F.max_pool2d(x, 2)))
        return x

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Penultimate representation (global-average-pooled conv3), shape (N, 64)."""
        return self._trunk(x).mean(dim=(2, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.embed(x))

    @torch.no_grad()
    def feature_maps(self, x: torch.Tensor, layer: str) -> torch.Tensor:
        if layer not in LAYER_NAMES:
            raise ValueError(f"layer must be one of {LAYER_NAMES}")
        return self._trunk(x, upto=layer)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
