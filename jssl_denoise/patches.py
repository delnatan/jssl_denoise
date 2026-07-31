from __future__ import annotations

import numpy as np
import torch


def sample_patch_batch(
    frames: list[np.ndarray],
    tile_size: int,
    n_tiles: int,
    rng: np.random.Generator,
    augment: bool = True,
) -> torch.Tensor:
    """Samples `n_tiles` random `tile_size x tile_size` crops from `frames`
    (normalized float32 2D arrays, possibly of differing sizes), with
    independent random H-flip / V-flip / 90-180-270-rotation augmentation
    per tile. Returns a (n_tiles, 1, tile_size, tile_size) float32 tensor.
    """
    tiles = np.empty((n_tiles, tile_size, tile_size), dtype=np.float32)

    for i in range(n_tiles):
        frame = frames[int(rng.integers(0, len(frames)))]
        height, width = frame.shape
        if height < tile_size or width < tile_size:
            raise ValueError(
                f"frame of shape {frame.shape} is smaller than tile_size={tile_size}"
            )
        top = int(rng.integers(0, height - tile_size + 1))
        left = int(rng.integers(0, width - tile_size + 1))
        tile = frame[top : top + tile_size, left : left + tile_size]

        if augment:
            if rng.random() < 0.5:
                tile = tile[::-1, :]
            if rng.random() < 0.5:
                tile = tile[:, ::-1]
            if rng.random() < 0.5:
                tile = np.rot90(tile, k=int(rng.integers(1, 4)))

        tiles[i] = tile

    return torch.from_numpy(tiles).unsqueeze(1)
