import torch

from jssl_denoise.losses import gaussian_nll_masked


def test_loss_only_depends_on_masked_pixels():
    torch.manual_seed(0)
    y = torch.rand(1, 1, 8, 8, requires_grad=True)
    mu = torch.rand(1, 1, 8, 8)
    sigma = torch.rand(1, 1, 8, 8) + 0.5

    mask = torch.zeros(8, 8, dtype=torch.bool)
    mask[2, 3] = True
    mask[5, 5] = True

    loss = gaussian_nll_masked(y, mu, sigma, mask)
    (grad,) = torch.autograd.grad(loss, y)

    nonzero = grad.abs() > 0
    expected = mask.view(1, 1, 8, 8)
    assert torch.equal(nonzero, expected.expand_as(nonzero))


def test_loss_matches_manual_formula_at_masked_pixel():
    y = torch.tensor([[[[2.0, 0.0], [0.0, 0.0]]]])
    mu = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])
    sigma = torch.tensor([[[[2.0, 1.0], [1.0, 1.0]]]])
    mask = torch.tensor([[True, False], [False, False]])

    loss = gaussian_nll_masked(y, mu, sigma, mask)

    expected = torch.log(torch.tensor(4.0)) + ((2.0 - 1.0) / 2.0) ** 2
    assert torch.isclose(loss, expected)
