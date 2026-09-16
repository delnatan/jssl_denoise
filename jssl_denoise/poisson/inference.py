from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..checkpoint import load_checkpoint
from ..inference import _DIHEDRAL_PARAMS
from ..masking import apply_masking
from ..normalization import RobustNormalizer
from ..training import _select_device
from .camera import CameraModel
from .likelihood import posterior_moments
from .network import GammaPriorNet
from .training import CHECKPOINT_METHOD

# Pixels per posterior_moments call; bounds the (pixels x window) buffers.
_POSTERIOR_CHUNK = 1 << 20


class PoissonDenoiser:
    """Bayesian denoising with a trained GammaPriorNet: the blind-spot prior
    is combined with each pixel's own observed value through the
    Poisson-Gaussian likelihood, and the posterior mean is returned.

    The prior must not see the pixel it is combined with, or that value
    would be counted twice. So the prior is built from `mask_spacing^2`
    masked passes, one per grid phase, each contributing only at its own
    masked pixels -- the same masking the network was trained on.
    """

    def __init__(
        self,
        prior_net: GammaPriorNet,
        camera: CameraModel,
        normalizer: RobustNormalizer,
        device: torch.device,
    ):
        self.prior_net = prior_net.to(device).eval()
        self.camera = camera
        self.normalizer = normalizer
        self.device = device

    @classmethod
    def from_checkpoint(
        cls, ckpt: dict, device: torch.device
    ) -> "PoissonDenoiser":
        if ckpt.get("method") != CHECKPOINT_METHOD:
            raise ValueError(
                "not a Poisson denoiser checkpoint (method="
                f"{ckpt.get('method')!r}); use jssl_denoise.Denoiser instead"
            )
        prior_net = GammaPriorNet(
            electrons_per_unit=1.0,  # restored from the state dict buffer
            base_filters=ckpt["config"].get("base_filters", 64),
        )
        prior_net.load_state_dict(ckpt["prior_net_state_dict"])
        return cls(
            prior_net,
            CameraModel.from_dict(ckpt["camera"]),
            RobustNormalizer.from_dict(ckpt["normalizer"]),
            device,
        )

    @classmethod
    def load(
        cls, path: str | Path, device: str | None = None
    ) -> "PoissonDenoiser":
        return cls.from_checkpoint(
            load_checkpoint(path), _select_device(device)
        )

    def denoise(
        self, image: np.ndarray, tta: bool = False, mask_spacing: int = 3
    ) -> tuple[np.ndarray, np.ndarray]:
        """image: 2D raw-ADU array, any size. Returns (denoised,
        posterior_std), float32 (H, W) arrays in ADU: the posterior mean of
        the noise-free signal offset + gain * lambda, and its posterior
        standard deviation.

        Costs `mask_spacing^2` network passes, times 8 with `tta`, which
        averages the posterior moments over the dihedral transforms of the
        prior (the observed pixel values are never transformed).
        """
        x = self._to_tensor(self.normalizer.transform(image))
        x_e = self._to_tensor(
            self.camera.to_electrons(np.asarray(image, dtype=np.float32))
        )

        transforms = _DIHEDRAL_PARAMS if tta else [(False, 0)]
        mean_sum = torch.zeros_like(x_e)
        var_sum = torch.zeros_like(x_e)
        with torch.no_grad():
            for flip, k in transforms:
                xt = torch.flip(x, dims=[-1]) if flip else x
                xt = torch.rot90(xt, k, dims=(-2, -1))
                s, r = self._blind_spot_prior(xt, mask_spacing)
                s, r = (
                    _undo_dihedral(t, flip, k) for t in (s, r)
                )
                mean, var = self._posterior(x_e, s, r)
                mean_sum += mean
                var_sum += var

        n = len(transforms)
        gain = self.camera.gain
        denoised = self.camera.to_adu(mean_sum / n)
        posterior_std = gain * torch.sqrt(var_sum / n)
        return _to_numpy(denoised), _to_numpy(posterior_std)

    def prior(
        self, image: np.ndarray, mask_spacing: int = 3
    ) -> tuple[np.ndarray, np.ndarray]:
        """Blind-spot Gamma prior (shape s, rate r) per pixel, in
        photoelectrons, for inspection. The prior mean is s / r."""
        x = self._to_tensor(self.normalizer.transform(image))
        with torch.no_grad():
            s, r = self._blind_spot_prior(x, mask_spacing)
        return _to_numpy(s), _to_numpy(r)

    def _blind_spot_prior(
        self, x: torch.Tensor, spacing: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        height, width = x.shape[-2:]
        s_map = torch.empty_like(x)
        r_map = torch.empty_like(x)
        for dy in range(spacing):
            for dx in range(spacing):
                mask = torch.zeros(
                    height, width, dtype=torch.bool, device=x.device
                )
                mask[dy::spacing, dx::spacing] = True
                masked, _ = apply_masking(x, mask)
                s, r = self.prior_net(masked)
                s_map[..., mask] = s[..., mask]
                r_map[..., mask] = r[..., mask]
        return s_map, r_map

    def _posterior(
        self, x_e: torch.Tensor, s: torch.Tensor, r: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        read_var_e = self.camera.read_var_e
        flat = [t.reshape(-1) for t in (x_e, s, r)]
        means, variances = [], []
        for start in range(0, flat[0].numel(), _POSTERIOR_CHUNK):
            chunk = [t[start : start + _POSTERIOR_CHUNK] for t in flat]
            mean, var = posterior_moments(*chunk, read_var_e)
            means.append(mean)
            variances.append(var)
        return (
            torch.cat(means).reshape(x_e.shape),
            torch.cat(variances).reshape(x_e.shape),
        )

    def _to_tensor(self, array: np.ndarray) -> torch.Tensor:
        return (
            torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.device)
        )


def _undo_dihedral(t: torch.Tensor, flip: bool, k: int) -> torch.Tensor:
    t = torch.rot90(t, -k, dims=(-2, -1))
    return torch.flip(t, dims=[-1]) if flip else t


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
