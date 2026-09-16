from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..networks import DNet

_MIN_PRIOR_MEAN_E = 1e-4
_MIN_SHAPE = 1e-3


class GammaPriorNet(nn.Module):
    """Maps a normalized (masked) image to a per-pixel Gamma(s, r) prior over
    the photon rate in photoelectrons.

    A 2-channel DNet predicts the prior mean m (channel 0) and shape s
    (channel 1); r = s / m. The mean is scaled by `electrons_per_unit`
    (RobustNormalizer.scale / gain), so a unit network output is about the
    95th-percentile signal regardless of the camera's gain -- the net sees
    and predicts on the same scale. That factor is a buffer, so a checkpoint
    carries it.
    """

    def __init__(self, electrons_per_unit: float, base_filters: int = 64):
        super().__init__()
        self.backbone = DNet(out_channels=2, base_filters=base_filters)
        self.register_buffer(
            "electrons_per_unit", torch.tensor(float(electrons_per_unit))
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, 1, H, W) normalized image. Returns (s, r), each (B, 1, H, W)."""
        out = self.backbone(x)
        mean = (
            F.softplus(out[:, :1]) * self.electrons_per_unit + _MIN_PRIOR_MEAN_E
        )
        shape = F.softplus(out[:, 1:]) + _MIN_SHAPE
        return shape, shape / mean
