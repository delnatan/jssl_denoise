from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def _inverse_softplus(y: float) -> float:
    return math.log(math.expm1(y))


class ConvBlock(nn.Module):
    """3x3 zero-padded conv + ReLU, preserving spatial size."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode="zeros")
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class UpsampleConv(nn.Module):
    """Nearest-neighbor 2x upsample followed by a 2x2 conv, used in place of
    a transposed convolution to avoid checkerboard artifacts (Section 4.1).
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = F.pad(x, (0, 1, 0, 1))  # asymmetric pad so a 2x2/stride-1 conv preserves the doubled size
        return self.conv(x)


class DNet(nn.Module):
    """Denoiser network: a 2-level U-Net with constant channel width,
    zero-padded convs, and nearest-neighbor+conv upsampling (Section 4.1 /
    Appendix A.1). Fully convolutional — accepts any (even or odd) H, W.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_filters: int = 64):
        super().__init__()
        f = base_filters

        self.enc1 = ConvBlock(in_channels, f)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(f, f)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(f, f)

        self.up2 = UpsampleConv(f, f)
        self.dec2 = ConvBlock(2 * f, f)
        self.up1 = UpsampleConv(f, f)
        self.dec1 = ConvBlock(2 * f, f)

        self.head = nn.Sequential(
            nn.Conv2d(f, f, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f, f, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.out_conv = nn.Conv2d(f, out_channels, kernel_size=1)  # linear output, no activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        pad_h = (-height) % 4
        pad_w = (-width) % 4
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))

        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        out = self.out_conv(self.head(d1))
        return out[..., :height, :width]


class NNet(nn.Module):
    """Noise network: the affine Poisson+Gaussian noise-variance model used
    by CMOS/sCMOS photon-transfer-curve calibration,

        sigma(mu)^2 = softplus(gain) * softplus(mu) + softplus(read_noise_var)

    with `gain` and `read_noise_var` as the only two learnable parameters,
    shared across all pixels (not a per-pixel network despite the name, kept
    for continuity with D-Net/N-Net terminology). `mu` is D-Net's predicted
    denoised value in the RobustNormalizer's linearly-scaled units, so this
    affine form is preserved from the raw-intensity affine noise model
    (Var(raw) = gain_raw*raw + read_var_raw) under that linear rescaling.
    `softplus(mu)` keeps the signal-dependent (Poisson) term non-negative
    and smooth as mu dips near/below the normalizer's background baseline
    (mu=0), instead of a plain affine `gain*mu` term going negative there;
    `softplus(read_noise_var)` keeps the read-noise floor strictly positive
    so sigma > 0 everywhere.
    """

    def __init__(self, init_gain: float = 1.0, init_read_noise_var: float = 0.1):
        super().__init__()
        self.raw_gain = nn.Parameter(torch.tensor(_inverse_softplus(init_gain)))
        self.raw_read_noise_var = nn.Parameter(
            torch.tensor(_inverse_softplus(init_read_noise_var))
        )

    def forward(self, mu: torch.Tensor) -> torch.Tensor:
        gain = F.softplus(self.raw_gain)
        read_noise_var = F.softplus(self.raw_read_noise_var)
        variance = gain * F.softplus(mu) + read_noise_var
        return torch.sqrt(variance)
