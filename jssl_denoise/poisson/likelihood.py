"""Poisson-Gaussian likelihood under a Gamma prior on the photon rate.

Per pixel, lambda ~ Gamma(s, r) (shape, rate), the photoelectron count
n | lambda ~ Poisson(lambda), and the observation in electrons is
x = n + eps with eps ~ N(0, read_var). Integrating out lambda makes n
negative-binomial (de Wolf et al., ISBI 2026), so

    p(x | s, r) = sum_n NB(n; s, r) * N(x - n; 0, read_var).

The Gaussian factor confines the sum to n within a few read-noise standard
deviations of x, so it is evaluated exactly over a window of 2W+1 counts
whose width depends only on read_var -- not on brightness. Given n, the
posterior over lambda is Gamma(s + n, r + 1), which gives the posterior
moments in closed form once E[n | x] and Var[n | x] are known.
"""

from __future__ import annotations

import math

import torch

# Window half-width in read-noise standard deviations; terms beyond it are
# below exp(-18) relative to the peak.
_WINDOW_STDS = 6.0


def log_marginal(
    x_e: torch.Tensor, s: torch.Tensor, r: torch.Tensor, read_var_e: float
) -> torch.Tensor:
    """log p(x | s, r), elementwise. x_e, s, r broadcast to a common shape;
    s, r > 0; read_var_e > 0 is the additive Gaussian variance in e^2."""
    _, log_joint = _log_joint_over_counts(x_e, s, r, read_var_e)
    return torch.logsumexp(log_joint, dim=-1)


def posterior_moments(
    x_e: torch.Tensor, s: torch.Tensor, r: torch.Tensor, read_var_e: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Posterior mean and variance of lambda given x, elementwise.

    With n | x marginalized over the window:
        E[lambda | x]   = (s + E[n|x]) / (r + 1)
        Var[lambda | x] = (s + E[n|x] + Var[n|x]) / (r + 1)^2
    (law of total variance over lambda | n ~ Gamma(s + n, r + 1)). As
    read_var -> 0 with integer x this is the paper's (s + x) / (r + 1).
    """
    n, log_joint = _log_joint_over_counts(x_e, s, r, read_var_e)
    weights = torch.softmax(log_joint, dim=-1)
    n_mean = (weights * n).sum(dim=-1)
    n_var = ((weights * n**2).sum(dim=-1) - n_mean**2).clamp_min(0.0)
    s, r = torch.broadcast_tensors(s, r)
    mean = (s + n_mean) / (r + 1)
    var = (s + n_mean + n_var) / (r + 1) ** 2
    return mean, var


def _log_joint_over_counts(
    x_e: torch.Tensor, s: torch.Tensor, r: torch.Tensor, read_var_e: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (n, log NB(n; s, r) + log N(x - n; 0, read_var)) with a
    trailing window dimension of 2W+1 counts.

    The window starts at max(round(x) - W, 0) and is shifted rather than
    clamped, so no count is included twice when x is near or below zero.
    """
    x_e, s, r = torch.broadcast_tensors(x_e, s, r)
    half_width = math.ceil(_WINDOW_STDS * math.sqrt(read_var_e))
    start = (torch.round(x_e.detach()) - half_width).clamp_min(0.0)
    offsets = torch.arange(
        2 * half_width + 1, device=x_e.device, dtype=x_e.dtype
    )
    n = start.unsqueeze(-1) + offsets

    s_, r_, x_ = s.unsqueeze(-1), r.unsqueeze(-1), x_e.unsqueeze(-1)
    log_nb = (
        torch.lgamma(s_ + n)
        - torch.lgamma(s_)
        - torch.lgamma(n + 1)
        - s_ * torch.log1p(1.0 / r_)
        - n * torch.log1p(r_)
    )
    log_gauss = -0.5 * (x_ - n) ** 2 / read_var_e - 0.5 * math.log(
        2 * math.pi * read_var_e
    )
    return n, log_nb + log_gauss
