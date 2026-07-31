from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

# 3x3 Gaussian (sigma=1) weights over the 8 direct neighbors, center excluded
# and renormalized to sum to 1, per Section 4.3 of Ollion et al. 2021.
_ORTHOGONAL_WEIGHT = math.exp(-0.5)  # distance 1
_DIAGONAL_WEIGHT = math.exp(-1.0)  # distance sqrt(2)
_raw_kernel = torch.tensor(
    [
        [_DIAGONAL_WEIGHT, _ORTHOGONAL_WEIGHT, _DIAGONAL_WEIGHT],
        [_ORTHOGONAL_WEIGHT, 0.0, _ORTHOGONAL_WEIGHT],
        [_DIAGONAL_WEIGHT, _ORTHOGONAL_WEIGHT, _DIAGONAL_WEIGHT],
    ]
)
NEIGHBOR_KERNEL = (_raw_kernel / _raw_kernel.sum()).view(1, 1, 3, 3)


def sample_grid_spacing(rng: np.random.Generator, low: int = 3, high: int = 5) -> int:
    """Draws an integer masking-grid spacing uniformly from {low, ..., high}."""
    return int(rng.integers(low, high + 1))


def make_grid_mask(shape: tuple[int, int], spacing: int, rng: np.random.Generator) -> np.ndarray:
    """Boolean (H, W) mask, True at masked grid points, with a random phase offset."""
    height, width = shape
    row_offset = int(rng.integers(0, spacing))
    col_offset = int(rng.integers(0, spacing))
    mask = np.zeros((height, width), dtype=bool)
    mask[row_offset::spacing, col_offset::spacing] = True
    return mask


def gaussian_neighbor_average(image: torch.Tensor) -> torch.Tensor:
    """Fixed (non-trainable) depthwise conv: replaces each pixel with a
    Gaussian-weighted average of its 8 direct neighbors. image: (B, 1, H, W).
    """
    kernel = NEIGHBOR_KERNEL.to(dtype=image.dtype, device=image.device)
    padded = F.pad(image, (1, 1, 1, 1), mode="reflect")
    return F.conv2d(padded, kernel)


def apply_masking(image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Replaces masked pixels with their neighbor-averaged value g(Y).

    image: (B, 1, H, W) tensor. mask: (H, W) bool tensor, shared across the batch
    (a single random grid is drawn per training step, per Fig. 1 of the paper).
    Returns (masked_image, g_values), both (B, 1, H, W); g_values is also used
    as the "central pixel" stand-in fed to N-Net's loss at masked locations.
    """
    g_values = gaussian_neighbor_average(image)
    mask_4d = mask.to(device=image.device).view(1, 1, *mask.shape)
    masked_image = torch.where(mask_4d, g_values, image)
    return masked_image, g_values
