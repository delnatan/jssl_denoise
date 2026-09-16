import numpy as np
import pytest


def simulate_stack(
    rng: np.random.Generator,
    n_frames: int = 3,
    size: int = 256,
    gain: float = 2.0,
    offset: float = 100.0,
    read_std: float = 1.2,
    background: float = 0.5,
) -> np.ndarray:
    """uint16 (T, H, W) raw-ADU stack: smooth blobs on a gradient, Poisson
    photon counts, Gaussian read noise, and ADU quantization."""
    yy, xx = np.mgrid[:size, :size]
    frames = []
    for _ in range(n_frames):
        lam = background + 30.0 * xx / size
        for _ in range(6):
            cy, cx = rng.uniform(0, size, 2)
            width = rng.uniform(size / 16, size / 8)
            lam = lam + rng.uniform(20, 150) * np.exp(
                -((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * width**2)
            )
        electrons = rng.poisson(lam) + rng.normal(0.0, read_std, lam.shape)
        frames.append(np.round(offset + gain * electrons).astype(np.uint16))
    return np.stack(frames)


@pytest.fixture
def poisson_gaussian_stack():
    return simulate_stack
