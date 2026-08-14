from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


def _pool_pixels(images: np.ndarray | Sequence[np.ndarray]) -> np.ndarray:
    if isinstance(images, np.ndarray):
        return images.ravel()
    return np.concatenate([np.asarray(im).ravel() for im in images])


@dataclass
class RobustNormalizer:
    """Maps raw pixel values to a robust ``(x - mode) / (p95 - mode)`` scale.

    Fluorescence microscopy images typically have a large, flat background
    (the mode) and a heavy tail of sparse bright signal, so the modal value
    and a high percentile make a more stable center/scale than mean/std.
    """

    mode: float
    scale: float

    @classmethod
    def fit(
        cls, images: np.ndarray | Sequence[np.ndarray], percentile: float = 95.0
    ) -> "RobustNormalizer":
        pooled = _pool_pixels(images).astype(np.float64)
        lo, hi = float(pooled.min()), float(pooled.max())
        p = float(np.percentile(pooled, percentile))

        if hi <= lo:
            mode = lo
        else:
            counts, edges = np.histogram(pooled, bins=4096, range=(lo, hi))
            peak = int(np.argmax(counts))
            mode = 0.5 * (edges[peak] + edges[peak + 1])

        scale = p - mode
        if scale <= 0:
            scale = 1.0

        return cls(mode=float(mode), scale=float(scale))

    def transform(self, image: np.ndarray) -> np.ndarray:
        return (
            (np.asarray(image, dtype=np.float64) - self.mode) / self.scale
        ).astype(np.float32)

    def inverse_transform(self, image: np.ndarray) -> np.ndarray:
        return (
            np.asarray(image, dtype=np.float64) * self.scale + self.mode
        ).astype(np.float32)

    def to_dict(self) -> dict:
        return {"mode": self.mode, "scale": self.scale}

    @classmethod
    def from_dict(cls, d: dict) -> "RobustNormalizer":
        return cls(mode=d["mode"], scale=d["scale"])
