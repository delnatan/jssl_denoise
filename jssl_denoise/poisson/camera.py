from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

# Variance of d = x - mean(4-neighbours) is 1.25 * Var(x) for white noise.
_RESIDUAL_VARIANCE_FACTOR = 1.25
_BLOCK = 8
_MIN_BLOCKS_PER_BIN = 20
_MAX_BINS = 50


@dataclass
class CameraModel:
    """Photon-transfer model of a camera pixel,

        x_ADU = offset + gain * (n + eps),  n ~ Poisson(lambda),
                                            eps ~ N(0, read_std^2),

    with `offset` in ADU, `gain` in ADU per photoelectron and `read_std` in
    photoelectrons.
    """

    offset: float
    gain: float
    read_std: float

    def to_electrons(self, adu):
        return (adu - self.offset) / self.gain

    def to_adu(self, electrons):
        return electrons * self.gain + self.offset

    @property
    def read_var_e(self) -> float:
        """Additive Gaussian variance in e^2: read noise plus the 1/12 ADU^2
        quantization of the digitizer. The quantization term is a physical
        floor that also keeps the likelihood's Gaussian width strictly
        positive when the fitted read noise comes out as zero."""
        return self.read_std**2 + 1.0 / (12.0 * self.gain**2)

    def to_dict(self) -> dict:
        return {
            "offset": self.offset,
            "gain": self.gain,
            "read_std": self.read_std,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CameraModel":
        return cls(offset=d["offset"], gain=d["gain"], read_std=d["read_std"])


def estimate_camera_model(
    images: np.ndarray | Sequence[np.ndarray],
    offset: float | None = None,
    gain: float | None = None,
) -> CameraModel:
    """Fits a CameraModel to noisy images alone (no dark or flat frames).

    Each frame's white-noise variance is measured from the residual against
    its 4-neighbour mean, which cancels locally linear signal, in 8x8 blocks.
    Blocks are binned by mean intensity; the per-bin median block variance is
    robust to blocks straddling edges or structure. A weighted line
    Var = gain * mean + b is then fitted across bins.

    The line only determines `gain` and the intercept b = read_var -
    gain * offset: offset and read noise trade off exactly. So the returned
    `read_std` is physically meaningful only when `offset` is supplied (e.g.
    the camera's nominal black level). Without it, the offset defaults to the
    1st percentile of block means -- the darkest regions taken as zero
    photons. With real background light that overestimates the offset, and
    `read_std` absorbs those background photons as extra Gaussian noise; on
    very dim data, block-mean noise and fixed-pattern offset variation can
    instead put it slightly below the nominal black level.

    `gain` (ADU/e-) may also be supplied, in which case only b is fitted.
    Row/column-correlated noise is not modelled and inflates the variance
    only by its (typically small) per-pixel contribution.
    """
    means, variances = _block_statistics(_as_frames(images))
    if means.size < 2 * _MIN_BLOCKS_PER_BIN:
        raise ValueError(
            f"only {means.size} {_BLOCK}x{_BLOCK} blocks available; need at "
            f"least {2 * _MIN_BLOCKS_PER_BIN} to fit a noise model"
        )

    bin_mean, bin_var, bin_count = _bin_by_intensity(means, variances)
    slope, intercept = _fit_line(bin_mean, bin_var, bin_count, slope=gain)
    if slope <= 0:
        raise ValueError(
            f"fitted gain {slope:.4g} ADU/e- is not positive; the images may "
            "lack enough intensity range, or be pre-processed (not raw ADU)"
        )

    if offset is None:
        offset = float(np.percentile(means, 1.0))

    # The measured variance includes the 1/12 ADU^2 quantization, which
    # CameraModel.read_var_e adds back on its own.
    read_var_adu = max(intercept + slope * offset - 1.0 / 12.0, 0.0)
    return CameraModel(
        offset=float(offset),
        gain=float(slope),
        read_std=float(np.sqrt(read_var_adu) / slope),
    )


def _as_frames(images: np.ndarray | Sequence[np.ndarray]) -> list[np.ndarray]:
    if isinstance(images, np.ndarray):
        if images.ndim == 2:
            return [images]
        if images.ndim == 3:
            return list(images)
        raise ValueError(
            f"expected a 2D image or a (T,H,W) stack, got shape {images.shape}"
        )
    return [np.asarray(im) for im in images]


def _block_statistics(frames: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Per-8x8-block mean of x and sample variance of the neighbour residual
    (rescaled to Var(x)), over all frames."""
    all_means, all_vars = [], []
    for frame in frames:
        x = np.asarray(frame, dtype=np.float64)
        center = x[1:-1, 1:-1]
        neighbours = 0.25 * (x[:-2, 1:-1] + x[2:, 1:-1] + x[1:-1, :-2] + x[1:-1, 2:])
        residual = center - neighbours

        bh, bw = center.shape[0] // _BLOCK, center.shape[1] // _BLOCK
        if bh == 0 or bw == 0:
            continue
        crop = (slice(0, bh * _BLOCK), slice(0, bw * _BLOCK))

        def blocks(a: np.ndarray) -> np.ndarray:
            return a[crop].reshape(bh, _BLOCK, bw, _BLOCK).swapaxes(1, 2).reshape(-1, _BLOCK * _BLOCK)

        all_means.append(blocks(center).mean(axis=1))
        all_vars.append(blocks(residual).var(axis=1, ddof=1) / _RESIDUAL_VARIANCE_FACTOR)

    if not all_means:
        return np.empty(0), np.empty(0)
    return np.concatenate(all_means), np.concatenate(all_vars)


def _bin_by_intensity(
    means: np.ndarray, variances: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Equal-count intensity bins; per bin, the mean intensity and the median
    block variance, de-biased for the median of a scaled chi-square."""
    n_bins = int(min(_MAX_BINS, means.size // _MIN_BLOCKS_PER_BIN))
    order = np.argsort(means)
    # Median of chi2_k / k is ~ (1 - 2/(9k))^3 (Wilson-Hilferty).
    dof = _BLOCK * _BLOCK - 1
    median_bias = (1.0 - 2.0 / (9.0 * dof)) ** 3

    bin_mean, bin_var, bin_count = [], [], []
    for idx in np.array_split(order, n_bins):
        bin_mean.append(means[idx].mean())
        bin_var.append(np.median(variances[idx]) / median_bias)
        bin_count.append(idx.size)
    return np.array(bin_mean), np.array(bin_var), np.array(bin_count, dtype=np.float64)


def _fit_line(
    x: np.ndarray,
    y: np.ndarray,
    count: np.ndarray,
    slope: float | None = None,
    iterations: int = 10,
) -> tuple[float, float]:
    """Iteratively reweighted least squares for y = slope * x + intercept.
    A variance estimate's standard error scales with the variance itself, so
    each bin is weighted by count / fitted_variance^2. With `slope` given,
    only the intercept is fitted."""
    fitted = y.copy()
    fit_slope, intercept = 0.0, 0.0
    for _ in range(iterations):
        w = count / np.maximum(fitted, 1e-12) ** 2
        if slope is None:
            design = np.stack([x, np.ones_like(x)], axis=1)
            sw = np.sqrt(w)
            (fit_slope, intercept), *_ = np.linalg.lstsq(design * sw[:, None], y * sw, rcond=None)
        else:
            fit_slope = slope
            intercept = float(np.sum(w * (y - slope * x)) / np.sum(w))
        fitted = fit_slope * x + intercept
    return float(fit_slope), float(intercept)
