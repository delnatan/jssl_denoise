from __future__ import annotations

import torch


def gaussian_nll_masked(
    y: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    mask: torch.Tensor,
    beta: float = 0.0,
) -> torch.Tensor:
    """Masked heteroscedastic Gaussian negative log-likelihood (Section 3):

        loss_i = log(sigma_i^2) + ((y_i - mu_i) / sigma_i)^2

    averaged over pixels where `mask` is True. All tensors broadcast to a
    common (B, 1, H, W) shape; `mask` is (H, W) or broadcastable to it.

    `beta` applies the beta-NLL reweighting of Seitzer et al. 2022
    (arXiv:2203.09168, "On the Pitfalls of Heteroscedastic Uncertainty
    Estimation with Probabilistic Neural Networks"): each pixel's NLL is
    scaled by a *stop-gradient* factor sigma_i^(2*beta) before averaging.
    Left at the default beta=0, this is the exact NLL above, which lets
    N-Net freely shrink sigma to soak up loss without mu (D-Net's output)
    actually improving -- mu's implicit gradient is inverse-variance
    weighted, so it goes quiet wherever sigma is currently large, exactly
    where mu most needs correcting. beta=1 cancels that weighting so mu's
    gradient reduces to plain (mu - y), i.e. ordinary MSE, regardless of
    what sigma currently predicts; 0 < beta < 1 interpolates between the
    two. sigma itself is still fit by the (now piecewise-reweighted) NLL
    either way, so it remains a valid per-pixel noise estimate.
    """
    mask = mask.to(device=y.device)
    nll = torch.log(sigma**2) + ((y - mu) / sigma) ** 2
    if beta > 0:
        nll = nll * (sigma**2).detach() ** beta
    return nll[..., mask].mean()
