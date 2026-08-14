import numpy as np
import torch

from jssl_denoise.config import TrainingConfig
from jssl_denoise.inference import _DIHEDRAL_PARAMS, Denoiser
from jssl_denoise.networks import DNet, NNet
from jssl_denoise.normalization import RobustNormalizer


def _make_denoiser() -> Denoiser:
    normalizer = RobustNormalizer(mode=100.0, scale=10.0)
    return Denoiser(
        DNet(base_filters=8), NNet(), normalizer, torch.device("cpu")
    )


def test_tta_dihedral_transforms_are_involutions():
    x = torch.arange(20 * 30, dtype=torch.float32).reshape(1, 1, 20, 30)

    for flip, k in _DIHEDRAL_PARAMS:
        xt = torch.flip(x, dims=[-1]) if flip else x
        xt = torch.rot90(xt, k, dims=(-2, -1))

        restored = torch.rot90(xt, -k, dims=(-2, -1))
        if flip:
            restored = torch.flip(restored, dims=[-1])

        assert torch.equal(restored, x)


def test_denoise_output_shape_and_dtype():
    denoiser = _make_denoiser()
    image = (np.random.default_rng(0).poisson(lam=100, size=(37, 53))).astype(
        np.uint16
    )

    denoised, noise_std_map = denoiser.denoise(image, tta=True)

    assert denoised.shape == image.shape
    assert noise_std_map.shape == image.shape
    assert denoised.dtype == np.float32
    assert noise_std_map.dtype == np.float32
    assert np.all(noise_std_map > 0)


def test_checkpoint_roundtrip_gives_identical_output(tmp_path):
    from jssl_denoise.checkpoint import load_checkpoint, save_checkpoint

    d_net, n_net = DNet(base_filters=8), NNet()
    normalizer = RobustNormalizer(mode=100.0, scale=10.0)
    denoiser = Denoiser(d_net, n_net, normalizer, torch.device("cpu"))

    ckpt = {
        "version": "1.0",
        "d_net_state_dict": d_net.state_dict(),
        "n_net_state_dict": n_net.state_dict(),
        "normalizer": normalizer.to_dict(),
        "config": {**vars(TrainingConfig()), "base_filters": 8},
    }
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, ckpt)

    reloaded = Denoiser.load(path, device="cpu")

    image = (np.random.default_rng(1).poisson(lam=100, size=(24, 24))).astype(
        np.uint16
    )
    d1, s1 = denoiser.denoise(image, tta=False)
    d2, s2 = reloaded.denoise(image, tta=False)

    np.testing.assert_allclose(d1, d2, atol=1e-5)
    np.testing.assert_allclose(s1, s2, atol=1e-5)

    loaded_dict = load_checkpoint(path)
    assert loaded_dict["version"] == "1.0"


def test_from_checkpoint_matches_load(tmp_path):
    from jssl_denoise.checkpoint import save_checkpoint

    d_net, n_net = DNet(base_filters=8), NNet()
    normalizer = RobustNormalizer(mode=100.0, scale=10.0)

    ckpt = {
        "version": "1.0",
        "d_net_state_dict": d_net.state_dict(),
        "n_net_state_dict": n_net.state_dict(),
        "normalizer": normalizer.to_dict(),
        "config": {**vars(TrainingConfig()), "base_filters": 8},
    }
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, ckpt)

    from_dict = Denoiser.from_checkpoint(ckpt, torch.device("cpu"))
    from_file = Denoiser.load(path, device="cpu")

    image = (np.random.default_rng(2).poisson(lam=100, size=(24, 24))).astype(
        np.uint16
    )
    d1, s1 = from_dict.denoise(image, tta=False)
    d2, s2 = from_file.denoise(image, tta=False)

    np.testing.assert_allclose(d1, d2, atol=1e-5)
    np.testing.assert_allclose(s1, s2, atol=1e-5)
