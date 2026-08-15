from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Hyperparameters for training a D-Net/N-Net pair, defaulting to the
    values reported in Ollion et al. 2021 (arXiv:2102.08023), scaled down
    where noted for small example datasets.
    """

    lr: float = 4e-4
    lr_decay_factor: float = 0.5
    lr_decay_every_epochs: int = 30
    lr_floor: float = 1e-6

    epochs: int = 400
    steps_per_epoch: int = 200

    tile_size: int = 96
    tiles_per_batch: int = 100

    mask_spacing_low: int = 3
    mask_spacing_high: int = 5

    # beta-NLL reweighting (Seitzer et al. 2022) for the masked Gaussian NLL,
    # see losses.gaussian_nll_masked. 0.0 is the exact NLL (sigma can decouple
    # from mu actually improving); 1.0 makes mu's gradient behave like plain
    # MSE; 0.5 is a middle ground.
    nll_beta: float = 0.5

    augment: bool = True
    base_filters: int = 64

    device: str | None = "mps"
    seed: int | None = None

    # Best-checkpoint tracking / early stopping, keyed off mu_mse rather than
    # the raw NLL loss -- see Trainer.fit's docstring for why loss keeps
    # drifting after mu (D-Net) has converged, purely from N-Net's two
    # noise-variance parameters wandering, which makes it the wrong signal to
    # plateau-detect on. `early_stop_patience` is epochs with no mu_mse
    # improvement (beyond `early_stop_min_delta`) before training stops early;
    # None disables early stopping and runs the full `epochs` count. Either
    # way, Trainer.fit always returns the best-mu_mse epoch's weights, not
    # necessarily the final epoch's.
    early_stop_patience: int | None = None
    early_stop_min_delta: float = 0.0
