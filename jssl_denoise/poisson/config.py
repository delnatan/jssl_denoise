from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PoissonTrainingConfig:
    """Hyperparameters for training a GammaPriorNet. Training-schedule and
    masking fields mirror jssl_denoise.config.TrainingConfig; kept as a
    separate dataclass while the two methods evolve independently.
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

    augment: bool = True
    base_filters: int = 64

    device: str | None = "mps"
    seed: int | None = None

    # Camera overrides, see poisson.camera.estimate_camera_model. `offset`
    # (ADU black level) is what makes the fitted read noise physically
    # meaningful; `gain` (ADU/e-) skips the slope fit.
    offset: float | None = None
    gain: float | None = None

    # Best-checkpoint tracking / early stopping on the epoch-mean loss. The
    # camera parameters are frozen during training, so unlike the Ollion
    # trainer the NLL cannot drift from noise parameters wandering.
    early_stop_patience: int | None = None
    early_stop_min_delta: float = 0.0
