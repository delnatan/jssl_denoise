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
from .patches import FrameSampler, sample_patch_batch


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

    The returned checkpoint is always the best-mu_mse epoch's weights, not
    necessarily the final epoch's -- once D-Net (mu) converges, N-Net's two
    noise-variance parameters can keep drifting under further gradient steps
    (their own optimum shifts along with the batch-to-batch noise in the
    residual estimate, with nothing left to anchor it once mu stops moving),
    which raises the raw NLL loss long after mu_mse -- the actual denoising-
    quality signal -- has flattened out. Training the extra epochs is mostly
    wasted compute in that regime, hence `early_stop_patience`.
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
        # Lives for the whole fit() call, not reconstructed per epoch or per
        # step -- see FrameSampler's docstring for why.
        frame_sampler = FrameSampler(len(normalized_frames), rng)

        d_net = DNet(base_filters=cfg.base_filters).to(device)
        n_net = NNet().to(device)
        optimizer = optim.Adam(
            list(d_net.parameters()) + list(n_net.parameters()), lr=cfg.lr
        )

        best_mu_mse = float("inf")
        best_epoch = 0
        best_state: tuple[dict, dict] | None = None
        epochs_since_improvement = 0

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
                    state = best_state or self._snapshot(d_net, n_net)
                    return self._checkpoint(*state, normalizer, cfg)

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

                optimizer.zero_grad()
                mu = d_net(masked_input)
                sigma = n_net(mu)
                loss = gaussian_nll_masked(
                    batch, mu, sigma, mask, beta=cfg.nll_beta
                )
                loss.backward()
                optimizer.step()

                loss_value = float(loss.detach().cpu())
                epoch_loss_sum += loss_value
                with torch.no_grad():
                    # mu's MSE against the raw (unmasked) target is the actual
                    # denoising-quality signal; unlike `loss` it can't be
                    # driven down by N-Net alone shrinking sigma, so it's the
                    # metric to watch for whether D-Net is still improving.
                    epoch_mse_sum += float(
                        ((batch - mu) ** 2)[..., mask].mean().cpu()
                    )
                    epoch_sigma_sum += float(sigma[..., mask].mean().cpu())
                if callback is not None:
                    callback.on_step_end(
                        step, cfg.steps_per_epoch, epoch, loss_value
                    )

            mu_mse = epoch_mse_sum / cfg.steps_per_epoch
            if callback is not None:
                callback.on_epoch_end(
                    epoch, cfg.epochs, epoch_loss_sum / cfg.steps_per_epoch, lr
                )
                if hasattr(callback, "on_epoch_metrics"):
                    callback.on_epoch_metrics(
                        epoch,
                        {
                            "mu_mse": mu_mse,
                            "sigma_mean": epoch_sigma_sum / cfg.steps_per_epoch,
                        },
                    )

            if mu_mse < best_mu_mse - cfg.early_stop_min_delta:
                best_mu_mse = mu_mse
                best_epoch = epoch
                best_state = self._snapshot(d_net, n_net)
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

            if (
                cfg.early_stop_patience is not None
                and epochs_since_improvement >= cfg.early_stop_patience
            ):
                if callback is not None and hasattr(callback, "on_early_stop"):
                    callback.on_early_stop(epoch, best_epoch, best_mu_mse)
                break

        state = best_state or self._snapshot(d_net, n_net)
        return self._checkpoint(*state, normalizer, cfg)

    @staticmethod
    def _snapshot(d_net: DNet, n_net: NNet) -> tuple[dict, dict]:
        """CPU-resident clone of both networks' state dicts -- cloned so
        later in-place optimizer steps on `d_net`/`n_net` can't mutate a
        stashed "best so far" snapshot, and moved off-device so holding one
        aside for the rest of a long run doesn't pin extra GPU/MPS memory."""
        return (
            {k: v.detach().clone().cpu() for k, v in d_net.state_dict().items()},
            {k: v.detach().clone().cpu() for k, v in n_net.state_dict().items()},
        )

    @staticmethod
    def _checkpoint(
        d_net_state: dict,
        n_net_state: dict,
        normalizer: RobustNormalizer,
        config: TrainingConfig,
    ) -> dict[str, Any]:
        return {
            "version": "1.0",
            "d_net_state_dict": d_net_state,
            "n_net_state_dict": n_net_state,
            "normalizer": normalizer.to_dict(),
            "config": asdict(config),
        }
