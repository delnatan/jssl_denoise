from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict
from typing import Any

import numpy as np
import torch
from torch import optim

from .callbacks import TrainingCallback
from .config import TrainingConfig
from .losses import gaussian_nll_masked
from .masking import apply_masking, make_grid_mask, sample_grid_spacing
from .networks import DNet, NNet
from .normalization import RobustNormalizer
from .patches import sample_patch_batch


def _to_frame_list(
    images: np.ndarray | Sequence[np.ndarray],
) -> list[np.ndarray]:
    if isinstance(images, np.ndarray):
        if images.ndim == 2:
            return [images]
        if images.ndim == 3:
            return [images[i] for i in range(images.shape[0])]
        raise ValueError(
            f"expected a 2D image or a (T,H,W) stack, got shape {images.shape}"
        )
    return [np.asarray(im) for im in images]


def _select_device(device: str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Trainer:
    """Orchestrates D-Net/N-Net training: sampling augmented patches, drawing
    a random masking grid per step, and optimizing the masked Gaussian NLL
    (Sections 3-4 of Ollion et al. 2021). Numpy in, plain dict out — no Qt.
    """

    def __init__(self, config: TrainingConfig):
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

        frames = _to_frame_list(images)
        normalizer = RobustNormalizer.fit(frames)
        normalized_frames = [normalizer.transform(f) for f in frames]

        d_net = DNet(base_filters=cfg.base_filters).to(device)
        n_net = NNet().to(device)
        optimizer = optim.Adam(
            list(d_net.parameters()) + list(n_net.parameters()), lr=cfg.lr
        )

        lr = cfg.lr
        for epoch in range(1, cfg.epochs + 1):
            if epoch > 1 and (epoch - 1) % cfg.lr_decay_every_epochs == 0:
                lr = max(lr * cfg.lr_decay_factor, cfg.lr_floor)
                for group in optimizer.param_groups:
                    group["lr"] = lr

            epoch_loss_sum = 0.0
            epoch_mse_sum = 0.0
            epoch_sigma_sum = 0.0
            for step in range(1, cfg.steps_per_epoch + 1):
                if should_stop is not None and should_stop():
                    return self._checkpoint(d_net, n_net, normalizer, cfg)

                batch = sample_patch_batch(
                    normalized_frames,
                    cfg.tile_size,
                    cfg.tiles_per_batch,
                    rng,
                    augment=cfg.augment,
                ).to(device)

                spacing = sample_grid_spacing(
                    rng, cfg.mask_spacing_low, cfg.mask_spacing_high
                )
                mask = torch.from_numpy(
                    make_grid_mask(
                        (cfg.tile_size, cfg.tile_size), spacing, rng
                    )
                ).to(device)
                masked_input, _ = apply_masking(batch, mask)

                optimizer.zero_grad()
                mu = d_net(masked_input)
                sigma = n_net(mu)
                loss = gaussian_nll_masked(batch, mu, sigma, mask, beta=cfg.nll_beta)
                loss.backward()
                optimizer.step()

                loss_value = float(loss.detach().cpu())
                epoch_loss_sum += loss_value
                with torch.no_grad():
                    # mu's MSE against the raw (unmasked) target is the actual
                    # denoising-quality signal; unlike `loss` it can't be
                    # driven down by N-Net alone shrinking sigma, so it's the
                    # metric to watch for whether D-Net is still improving.
                    epoch_mse_sum += float(((batch - mu) ** 2)[..., mask].mean().cpu())
                    epoch_sigma_sum += float(sigma[..., mask].mean().cpu())
                if callback is not None:
                    callback.on_step_end(
                        step, cfg.steps_per_epoch, epoch, loss_value
                    )

            if callback is not None:
                callback.on_epoch_end(
                    epoch, cfg.epochs, epoch_loss_sum / cfg.steps_per_epoch, lr
                )
                if hasattr(callback, "on_epoch_metrics"):
                    callback.on_epoch_metrics(
                        epoch,
                        {
                            "mu_mse": epoch_mse_sum / cfg.steps_per_epoch,
                            "sigma_mean": epoch_sigma_sum / cfg.steps_per_epoch,
                        },
                    )

        return self._checkpoint(d_net, n_net, normalizer, cfg)

    @staticmethod
    def _checkpoint(
        d_net: DNet,
        n_net: NNet,
        normalizer: RobustNormalizer,
        config: TrainingConfig,
    ) -> dict[str, Any]:
        return {
            "version": "1.0",
            "d_net_state_dict": d_net.state_dict(),
            "n_net_state_dict": n_net.state_dict(),
            "normalizer": normalizer.to_dict(),
            "config": asdict(config),
        }
