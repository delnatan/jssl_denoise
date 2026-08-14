from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import torch

from .checkpoint import load_checkpoint
from .networks import DNet, NNet
from .normalization import RobustNormalizer
from .training import _select_device

_DIHEDRAL_PARAMS = list(
    itertools.product([False, True], range(4))
)  # (flip, rot90_k)


class Denoiser:
    """Runs a trained D-Net/N-Net pair on full-size images, with optional
    test-time augmentation (Section 4.1: averaging predictions over the 8
    dihedral flip/rotation transforms). Plain numpy in, plain numpy out.
    """

    def __init__(
        self,
        d_net: DNet,
        n_net: NNet,
        normalizer: RobustNormalizer,
        device: torch.device,
    ):
        self.d_net = d_net.to(device).eval()
        self.n_net = n_net.to(device).eval()
        self.normalizer = normalizer
        self.device = device

    @classmethod
    def from_checkpoint(cls, ckpt: dict, device: torch.device) -> "Denoiser":
        """Build a Denoiser from an already-loaded checkpoint dict (e.g. the
        one `Trainer.fit` returns directly, with no save/load roundtrip).
        """
        base_filters = ckpt["config"].get("base_filters", 64)

        d_net = DNet(base_filters=base_filters)
        d_net.load_state_dict(ckpt["d_net_state_dict"])
        n_net = NNet()
        n_net.load_state_dict(ckpt["n_net_state_dict"])
        normalizer = RobustNormalizer.from_dict(ckpt["normalizer"])

        return cls(d_net, n_net, normalizer, device)

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "Denoiser":
        return cls.from_checkpoint(
            load_checkpoint(path), _select_device(device)
        )

    def denoise(
        self, image: np.ndarray, tta: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """image: 2D array, any size. Returns (denoised, noise_std_map), both
        float32 (H, W) arrays in the original intensity units.
        """
        normalized = self.normalizer.transform(image)
        x = (
            torch.from_numpy(normalized)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():
            mu = self._predict_denoised(x, tta=tta)
            sigma = self.n_net(mu)

        denoised = self.normalizer.inverse_transform(
            mu.squeeze(0).squeeze(0).cpu().numpy()
        )
        noise_std_map = (
            sigma.squeeze(0).squeeze(0).cpu().numpy() * self.normalizer.scale
        ).astype(np.float32)

        return denoised, noise_std_map

    def _predict_denoised(self, x: torch.Tensor, tta: bool) -> torch.Tensor:
        if not tta:
            return self.d_net(x)

        predictions = []
        for flip, k in _DIHEDRAL_PARAMS:
            xt = torch.flip(x, dims=[-1]) if flip else x
            xt = torch.rot90(xt, k, dims=(-2, -1))
            yt = self.d_net(xt)
            yt = torch.rot90(yt, -k, dims=(-2, -1))
            if flip:
                yt = torch.flip(yt, dims=[-1])
            predictions.append(yt)
        return torch.stack(predictions, dim=0).mean(dim=0)
