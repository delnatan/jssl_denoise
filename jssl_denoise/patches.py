from __future__ import annotations

import numpy as np
import torch


class FrameSampler:
    """Yields frame indices in shuffled order, one full shuffled pass at a
    time, reshuffling and refilling whenever the current pass is exhausted.

    Guarantees every frame is drawn once before any frame repeats -- unlike
    drawing a frame index uniformly with replacement each tile (plain
    `rng.integers(0, n_frames)`), which for a pool of frames small relative
    to steps_per_epoch * tiles_per_batch can by chance under- or over-sample
    a given frame for a long stretch of steps. One `FrameSampler` is meant
    to live for a whole `Trainer.fit` call (not reconstructed per epoch),
    so shuffled passes run continuously across epoch boundaries -- "epoch"
    here is just a grouping of steps for LR decay/logging, not a dataset
    pass, so there's no natural per-epoch reshuffle point anyway.
    """

    def __init__(self, n_frames: int, rng: np.random.Generator):
        self._n_frames = n_frames
        self._rng = rng
        self._order: list[int] = []

    def next(self) -> int:
        if not self._order:
            self._order = list(self._rng.permutation(self._n_frames))
        return self._order.pop()


def sample_patch_batch(
    frames: list[np.ndarray],
    tile_size: int,
    n_tiles: int,
    rng: np.random.Generator,
    sampler: FrameSampler,
    augment: bool = True,
) -> torch.Tensor:
    """Samples `n_tiles` random `tile_size x tile_size` crops from `frames`
    (normalized float32 2D arrays, possibly of differing sizes), with
    independent random H-flip / V-flip / 90-180-270-rotation augmentation
    per tile. Which frame each tile is cropped from comes from `sampler`
    (shuffled, even coverage) rather than a fresh uniform draw per tile.
    Returns a (n_tiles, 1, tile_size, tile_size) float32 tensor.
    """
    tiles = np.empty((n_tiles, tile_size, tile_size), dtype=np.float32)

    for i in range(n_tiles):
        frame = frames[sampler.next()]
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
