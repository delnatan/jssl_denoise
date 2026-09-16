from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict
from typing import Any

import numpy as np
import torch
from torch import optim

from ..callbacks import TrainingCallback
from ..masking import apply_masking, make_grid_mask, sample_grid_spacing
from ..normalization import RobustNormalizer
from ..patches import FrameSampler, sample_patch_batch
from ..training import _select_device, _to_frame_list
from .camera import CameraModel, estimate_camera_model
from .config import PoissonTrainingConfig
from .likelihood import log_marginal
from .network import GammaPriorNet

CHECKPOINT_METHOD = "poisson"
CHECKPOINT_VERSION = "1.0"


class PoissonTrainer:
    """Trains a GammaPriorNet as a blind-spot prior under the exact
    Poisson-Gaussian likelihood (de Wolf et al., ISBI 2026, extended with
    camera read noise).

    The camera model (offset, gain, read noise) is fitted from the images
    before training and held fixed: a free per-pixel prior variance can
    absorb any change in gain or read noise without changing the
    likelihood's first two moments, so they cannot be learned jointly.
    The network then only learns the prior, from masked pixels, with the
    negative log marginal likelihood as the loss.
    """

    def __init__(self, config: PoissonTrainingConfig):
        self.config = config

    def fit(
        self,
        images: np.ndarray | Sequence[np.ndarray],
        callback: TrainingCallback | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        cfg = self.config
        device = _select_device(cfg.device)
        rng = np.random.default_rng(cfg.seed)
        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)

        frames = _to_frame_list(images)
        camera = estimate_camera_model(frames, offset=cfg.offset, gain=cfg.gain)
        normalizer = RobustNormalizer.fit(frames)
        normalized_frames = [normalizer.transform(f) for f in frames]
        frame_sampler = FrameSampler(len(normalized_frames), rng)

        net = GammaPriorNet(
            electrons_per_unit=normalizer.scale / camera.gain,
            base_filters=cfg.base_filters,
        ).to(device)
        optimizer = optim.Adam(net.parameters(), lr=cfg.lr)
        read_var_e = camera.read_var_e

        best_loss = float("inf")
        best_epoch = 0
        best_state: dict | None = None
        epochs_since_improvement = 0

        lr = cfg.lr
        for epoch in range(1, cfg.epochs + 1):
            if epoch > 1 and (epoch - 1) % cfg.lr_decay_every_epochs == 0:
                lr = max(lr * cfg.lr_decay_factor, cfg.lr_floor)
                for group in optimizer.param_groups:
                    group["lr"] = lr

            epoch_loss_sum = 0.0
            epoch_prior_mse_sum = 0.0
            for step in range(1, cfg.steps_per_epoch + 1):
                if should_stop is not None and should_stop():
                    state = best_state or _snapshot(net)
                    return _checkpoint(state, camera, normalizer, cfg)

                batch = sample_patch_batch(
                    normalized_frames,
                    cfg.tile_size,
                    cfg.tiles_per_batch,
                    rng,
                    frame_sampler,
                    augment=cfg.augment,
                ).to(device)
                spacing = sample_grid_spacing(
                    rng, cfg.mask_spacing_low, cfg.mask_spacing_high
                )
                mask = torch.from_numpy(
                    make_grid_mask((cfg.tile_size, cfg.tile_size), spacing, rng)
                ).to(device)
                masked_input, _ = apply_masking(batch, mask)

                # Targets in photoelectrons, from the same augmented tiles.
                x_e = camera.to_electrons(
                    batch * normalizer.scale + normalizer.mode
                )

                optimizer.zero_grad()
                s, r = net(masked_input)
                x_m, s_m, r_m = x_e[..., mask], s[..., mask], r[..., mask]
                loss = -log_marginal(x_m, s_m, r_m, read_var_e).mean()
                loss.backward()
                optimizer.step()

                loss_value = float(loss.detach().cpu())
                epoch_loss_sum += loss_value
                with torch.no_grad():
                    epoch_prior_mse_sum += float(
                        ((s_m / r_m - x_m) ** 2).mean().cpu()
                    )
                if callback is not None:
                    callback.on_step_end(
                        step, cfg.steps_per_epoch, epoch, loss_value
                    )

            epoch_loss = epoch_loss_sum / cfg.steps_per_epoch
            if callback is not None:
                callback.on_epoch_end(epoch, cfg.epochs, epoch_loss, lr)
                if hasattr(callback, "on_epoch_metrics"):
                    callback.on_epoch_metrics(
                        epoch,
                        {
                            "prior_mse_e": epoch_prior_mse_sum
                            / cfg.steps_per_epoch
                        },
                    )

            if epoch_loss < best_loss - cfg.early_stop_min_delta:
                best_loss = epoch_loss
                best_epoch = epoch
                best_state = _snapshot(net)
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

            if (
                cfg.early_stop_patience is not None
                and epochs_since_improvement >= cfg.early_stop_patience
            ):
                if callback is not None and hasattr(callback, "on_early_stop"):
                    callback.on_early_stop(epoch, best_epoch, best_loss)
                break

        state = best_state or _snapshot(net)
        return _checkpoint(state, camera, normalizer, cfg)


def _snapshot(net: GammaPriorNet) -> dict:
    """Detached CPU clone, immune to later in-place optimizer steps."""
    return {k: v.detach().clone().cpu() for k, v in net.state_dict().items()}


def _checkpoint(
    net_state: dict,
    camera: CameraModel,
    normalizer: RobustNormalizer,
    config: PoissonTrainingConfig,
) -> dict[str, Any]:
    return {
        "method": CHECKPOINT_METHOD,
        "version": CHECKPOINT_VERSION,
        "prior_net_state_dict": net_state,
        "camera": camera.to_dict(),
        "normalizer": normalizer.to_dict(),
        "config": asdict(config),
    }
