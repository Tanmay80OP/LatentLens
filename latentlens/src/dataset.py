"""Procedural synthetic shape dataset with a controllable background shortcut.

Design notes
------------
* Class = object shape (circle / square / triangle). Object colour is achromatic
  (light or dark) and carries *no* class information.
* Each class has an associated background hue. The three hues are close to
  isoluminant, so hue (not brightness) is the shortcut cue.
* Object geometry (position, size, rotation, colour, label) depends only on
  ``(n, seed, size)``. Background assignment is drawn from a separate RNG. Two
  datasets with the same ``(n, seed)`` but different ``bg_mode`` therefore contain
  the *same objects on different backgrounds* -> a controlled comparison.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw

CLASSES = ["circle", "square", "triangle"]
N_CLASSES = len(CLASSES)

# Class-associated background colours (RGB in [0,1]); near-isoluminant by design.
BG_PALETTE = np.array(
    [
        [0.85, 0.25, 0.25],  # circle   -> red-ish
        [0.20, 0.62, 0.20],  # square   -> green-ish
        [0.28, 0.40, 0.85],  # triangle -> blue-ish
    ],
    dtype=np.float32,
)
BG_NAMES = ["red", "green", "blue"]

BG_MODES = ("correlated", "random", "reversed")
_BG_SALT = {"correlated": 101, "random": 202, "reversed": 303}

SUPERSAMPLE = 2


@dataclass
class SyntheticDataset:
    """Container for a generated dataset. ``x`` is float32 (N, 3, H, W) in [0, 1]."""

    x: np.ndarray
    y: np.ndarray  # true class (shape)
    bg: np.ndarray  # class implied by the background colour
    bg_mode: str
    seed: int

    def __len__(self) -> int:
        return len(self.y)


def luminance(rgb: np.ndarray) -> np.ndarray:
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def _polygon(cx: float, cy: float, radius: float, n_sides: int, angle: float):
    return [
        (
            cx + radius * np.cos(angle + 2 * np.pi * k / n_sides),
            cy + radius * np.sin(angle + 2 * np.pi * k / n_sides),
        )
        for k in range(n_sides)
    ]


@lru_cache(maxsize=32)
def _render_objects(n: int, seed: int, size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render object masks, object colours and labels. Depends only on (n, seed, size)."""
    rng = np.random.default_rng(seed)
    labels = np.arange(n) % N_CLASSES  # exactly balanced classes
    rng.shuffle(labels)
    big = size * SUPERSAMPLE
    masks = np.zeros((n, size, size), dtype=np.float32)
    colors = np.zeros((n, 3), dtype=np.float32)
    s = size / 32.0
    for i in range(n):
        label = int(labels[i])
        cx = (size / 2 + rng.uniform(-3, 3) * s) * SUPERSAMPLE
        cy = (size / 2 + rng.uniform(-3, 3) * s) * SUPERSAMPLE
        r = rng.uniform(6.5, 10.0) * s * SUPERSAMPLE
        angle = rng.uniform(0, 2 * np.pi)
        img = Image.new("L", (big, big), 0)
        d = ImageDraw.Draw(img)
        if label == 0:  # circle (slightly eccentric to avoid a perfectly clean template)
            ax, ay = r * rng.uniform(0.92, 1.08), r * rng.uniform(0.92, 1.08)
            d.ellipse([cx - ax, cy - ay, cx + ax, cy + ay], fill=255)
        elif label == 1:  # square (rotated)
            d.polygon(_polygon(cx, cy, r * 1.1, 4, angle + np.pi / 4), fill=255)
        else:  # equilateral triangle (rotated)
            d.polygon(_polygon(cx, cy, r * 1.15, 3, angle), fill=255)
        small = img.resize((size, size), Image.BOX)
        masks[i] = np.asarray(small, dtype=np.float32) / 255.0
        # achromatic object colour, light or dark, with a small tint (carries no class info)
        base = rng.uniform(0.86, 1.0) if rng.random() < 0.5 else rng.uniform(0.0, 0.14)
        colors[i] = np.clip(base + rng.normal(0, 0.03, 3), 0, 1)
    labels = labels.astype(np.int64)
    for a in (masks, colors, labels):
        a.setflags(write=False)  # cached arrays must never be mutated
    return masks, colors, labels


def assign_backgrounds(labels: np.ndarray, mode: str, correlation: float, rng: np.random.Generator) -> np.ndarray:
    """Return the class index whose background colour each image receives."""
    n = len(labels)
    if mode == "correlated":
        match = rng.random(n) < correlation
        other = (labels + rng.integers(1, N_CLASSES, n)) % N_CLASSES
        return np.where(match, labels, other).astype(np.int64)
    if mode == "random":  # background independent of the label
        return rng.integers(0, N_CLASSES, n).astype(np.int64)
    if mode == "reversed":  # background ALWAYS points to a wrong class
        return ((labels + rng.integers(1, N_CLASSES, n)) % N_CLASSES).astype(np.int64)
    raise ValueError(f"unknown bg_mode {mode!r}; expected one of {BG_MODES}")


def make_dataset(
    n: int,
    seed: int,
    bg_mode: str = "correlated",
    correlation: float = 0.9,
    size: int = 32,
    noise: float = 0.03,
) -> SyntheticDataset:
    """Generate ``n`` images. See module docstring for the controlled-comparison design."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0.0 <= correlation <= 1.0:
        raise ValueError("correlation must be in [0, 1]")
    if bg_mode not in BG_MODES:
        raise ValueError(f"unknown bg_mode {bg_mode!r}; expected one of {BG_MODES}")
    masks, obj_colors, labels = _render_objects(int(n), int(seed), int(size))
    rng = np.random.default_rng(int(seed) + _BG_SALT[bg_mode] * 7919)
    bg_idx = assign_backgrounds(labels, bg_mode, correlation, rng)
    bg_col = BG_PALETTE[bg_idx] + rng.normal(0, 0.04, (n, 3)).astype(np.float32)
    bg_col = bg_col * rng.uniform(0.92, 1.08, (n, 1)).astype(np.float32)
    bg_col = np.clip(bg_col, 0, 1)
    m = masks[:, None, :, :]
    x = (1 - m) * bg_col[:, :, None, None] + m * obj_colors[:, :, None, None]
    x = x + rng.normal(0, noise, x.shape).astype(np.float32)
    x = np.clip(x, 0, 1).astype(np.float32)
    return SyntheticDataset(x=x, y=labels.copy(), bg=bg_idx, bg_mode=bg_mode, seed=int(seed))
