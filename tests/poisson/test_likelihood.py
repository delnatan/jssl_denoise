import math

import numpy as np
import torch

from jssl_denoise.poisson.likelihood import log_marginal, posterior_moments


def _brute_force(x, s, r, read_var, n_max=2000):
    """float64 reference: explicit sum over n in [0, n_max]."""
    n = np.arange(n_max + 1, dtype=np.float64)
    lg = np.vectorize(math.lgamma)
    log_nb = (
        lg(s + n) - math.lgamma(s) - lg(n + 1)
        + s * math.log(r / (r + 1)) - n * math.log(r + 1)
    )
    log_g = -0.5 * (x - n) ** 2 / read_var - 0.5 * math.log(2 * math.pi * read_var)
    log_joint = log_nb + log_g
    peak = log_joint.max()
    w = np.exp(log_joint - peak)
    log_p = peak + math.log(w.sum())
    w /= w.sum()
    n_mean = (w * n).sum()
    n_var = (w * n**2).sum() - n_mean**2
    mean = (s + n_mean) / (r + 1)
    var = (s + n_mean + n_var) / (r + 1) ** 2
    return log_p, mean, var


def _t(*values):
    return [torch.tensor(v, dtype=torch.float64) for v in values]


def test_matches_brute_force_at_low_and_high_counts():
    for x, s, r, read_var in [
        (0.3, 2.0, 0.5, 1.0),
        (-2.5, 0.7, 3.0, 1.5),
        (12.4, 5.0, 0.4, 0.8),
        (480.0, 30.0, 0.06, 2.0),
    ]:
        ref_log_p, ref_mean, ref_var = _brute_force(x, s, r, read_var)
        tx, ts, tr = _t(x, s, r)
        mean, var = posterior_moments(tx, ts, tr, read_var)
        assert math.isclose(float(log_marginal(tx, ts, tr, read_var)), ref_log_p, abs_tol=1e-8)
        assert math.isclose(float(mean), ref_mean, rel_tol=1e-8)
        assert math.isclose(float(var), ref_var, rel_tol=1e-8)


def test_marginal_integrates_to_one():
    xs = torch.linspace(-15.0, 80.0, 20001, dtype=torch.float64)
    s, r = _t(3.0, 0.5)
    density = torch.exp(log_marginal(xs, s, r, 1.2))
    integral = torch.trapezoid(density, xs)
    assert abs(float(integral) - 1.0) < 1e-6


def test_reduces_to_negative_binomial_and_paper_estimate_without_read_noise():
    s, r, read_var = 2.5, 0.8, 1e-4
    for x in [0, 1, 4, 9]:
        tx, ts, tr = _t(float(x), s, r)
        log_nb = (
            math.lgamma(s + x) - math.lgamma(s) - math.lgamma(x + 1)
            + s * math.log(r / (r + 1)) - x * math.log(r + 1)
        )
        log_gauss_peak = -0.5 * math.log(2 * math.pi * read_var)
        assert math.isclose(
            float(log_marginal(tx, ts, tr, read_var)), log_nb + log_gauss_peak, abs_tol=1e-8
        )
        mean, _ = posterior_moments(tx, ts, tr, read_var)
        assert math.isclose(float(mean), (s + x) / (r + 1), rel_tol=1e-8)


def test_float32_gradients_finite_far_below_zero():
    x = torch.tensor([-40.0, -3.0, 0.0, 2.5, 1e4])
    s = torch.full_like(x, 1.5, requires_grad=True)
    r = torch.full_like(x, 0.2, requires_grad=True)
    loss = -log_marginal(x, s, r, 1.0).sum()
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(s.grad).all() and torch.isfinite(r.grad).all()
