import numpy as np
import pytest
import torch

from jssl_denoise.checkpoint import save_checkpoint
from jssl_denoise.normalization import RobustNormalizer
from jssl_denoise.poisson import (
    CameraModel,
    GammaPriorNet,
    PoissonDenoiser,
    PoissonTrainingConfig,
)
from jssl_denoise.poisson.training import _checkpoint, _snapshot


def _make_denoiser(seed: int = 0) -> PoissonDenoiser:
    torch.manual_seed(seed)
    camera = CameraModel(offset=100.0, gain=2.0, read_std=1.0)
    normalizer = RobustNormalizer(mode=110.0, scale=40.0)
    net = GammaPriorNet(electrons_per_unit=20.0, base_filters=8)
    return PoissonDenoiser(net, camera, normalizer, torch.device("cpu"))


def _image(seed: int, shape=(37, 53)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (100 + 2 * rng.poisson(8, size=shape)).astype(np.uint16)


def test_output_shape_dtype_and_positive_std():
    denoiser = _make_denoiser()
    image = _image(0)
    for tta in (False, True):
        denoised, posterior_std = denoiser.denoise(image, tta=tta)
        assert denoised.shape == image.shape
        assert posterior_std.shape == image.shape
        assert denoised.dtype == np.float32
        assert posterior_std.dtype == np.float32
        assert np.all(np.isfinite(denoised))
        assert np.all(posterior_std > 0)


@pytest.mark.parametrize("pixel", [(0, 0), (17, 20), (36, 52), (5, 51)])
def test_prior_is_blind_to_its_own_pixel(pixel):
    denoiser = _make_denoiser()
    image = _image(1)
    perturbed = image.copy()
    perturbed[pixel] = 4000

    s0, r0 = denoiser.prior(image)
    s1, r1 = denoiser.prior(perturbed)

    assert s0[pixel] == pytest.approx(s1[pixel], rel=1e-5)
    assert r0[pixel] == pytest.approx(r1[pixel], rel=1e-5)
    # ...while the perturbation does reach its neighbours' priors
    assert not np.allclose(s0, s1)


def test_bright_observation_pulls_posterior_above_prior():
    denoiser = _make_denoiser()
    image = _image(2)
    s, r = denoiser.prior(image)
    bright = image.copy()
    bright[10, 10] = 100 + 2 * 200  # far above any prior mean
    denoised, _ = denoiser.denoise(bright)
    prior_mean_adu = denoiser.camera.to_adu(s[10, 10] / r[10, 10])
    assert denoised[10, 10] > prior_mean_adu


def test_checkpoint_roundtrip_gives_identical_output(tmp_path):
    denoiser = _make_denoiser()
    config = PoissonTrainingConfig(base_filters=8)
    ckpt = _checkpoint(
        _snapshot(denoiser.prior_net),
        denoiser.camera,
        denoiser.normalizer,
        config,
    )
    path = tmp_path / "poisson.pt"
    save_checkpoint(path, ckpt)
    reloaded = PoissonDenoiser.load(path, device="cpu")

    image = _image(3, shape=(24, 24))
    d1, s1 = denoiser.denoise(image)
    d2, s2 = reloaded.denoise(image)
    np.testing.assert_allclose(d1, d2, atol=1e-5)
    np.testing.assert_allclose(s1, s2, atol=1e-5)


def test_rejects_ollion_checkpoint():
    ollion_ckpt = {"version": "1.0", "d_net_state_dict": {}, "config": {}}
    with pytest.raises(ValueError, match="not a Poisson denoiser checkpoint"):
        PoissonDenoiser.from_checkpoint(ollion_ckpt, torch.device("cpu"))
