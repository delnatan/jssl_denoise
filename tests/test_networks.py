import torch
from torch import nn

from jssl_denoise.networks import DNet, NNet


def test_dnet_preserves_shape_on_odd_size_input():
    net = DNet()
    x = torch.randn(1, 1, 97, 143)

    out = net(x)

    assert out.shape == x.shape
    assert out.dtype == x.dtype


def test_dnet_output_layer_has_no_activation():
    net = DNet()

    assert isinstance(net.out_conv, nn.Conv2d)
    # negative bias must survive unclipped to the output (rules out ReLU/sigmoid/exp on top)
    with torch.no_grad():
        net.out_conv.weight.zero_()
        net.out_conv.bias.fill_(-5.0)
    out = net(torch.randn(1, 1, 32, 32))
    assert torch.allclose(out, torch.full_like(out, -5.0))


def test_nnet_output_is_always_positive():
    net = NNet()
    mu = torch.randn(4, 1, 20, 20) * 100  # wide range of magnitudes

    sigma = net(mu)

    assert torch.all(sigma > 0)
    assert sigma.shape == mu.shape


def test_dnet_receptive_field_is_reasonably_local():
    net = DNet()
    net.eval()
    x = torch.zeros(1, 1, 64, 64, requires_grad=True)

    out = net(x)
    out[0, 0, 32, 32].backward()

    influenced = (x.grad.abs()[0, 0] > 0).nonzero()
    height_span = (influenced[:, 0].max() - influenced[:, 0].min()).item() + 1
    width_span = (influenced[:, 1].max() - influenced[:, 1].min()).item() + 1

    assert 5 <= height_span <= 60
    assert 5 <= width_span <= 60
