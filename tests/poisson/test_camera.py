import numpy as np
import pytest

from jssl_denoise.poisson.camera import CameraModel, estimate_camera_model


def test_recovers_gain_and_read_noise_with_known_offset(poisson_gaussian_stack):
    for seed in range(3):
        stack = poisson_gaussian_stack(np.random.default_rng(seed))
        camera = estimate_camera_model(stack, offset=100.0)
        assert camera.offset == 100.0
        assert camera.gain == pytest.approx(2.0, rel=0.05)
        assert camera.read_std == pytest.approx(1.2, rel=0.10)


def test_default_offset_absorbs_background_light(poisson_gaussian_stack):
    stack = poisson_gaussian_stack(np.random.default_rng(0))
    known = estimate_camera_model(stack, offset=100.0)
    estimated = estimate_camera_model(stack)
    # up to block-mean noise (~0.3 ADU here)
    assert estimated.offset >= 100.0 - 1.0
    assert estimated.read_std >= known.read_std


def test_gain_override_is_respected(poisson_gaussian_stack):
    stack = poisson_gaussian_stack(np.random.default_rng(1))
    camera = estimate_camera_model(stack, offset=100.0, gain=2.0)
    assert camera.gain == 2.0
    assert camera.read_std == pytest.approx(1.2, rel=0.10)


def test_accepts_single_frame_and_frame_list(poisson_gaussian_stack):
    stack = poisson_gaussian_stack(np.random.default_rng(2))
    from_list = estimate_camera_model(list(stack), offset=100.0)
    from_array = estimate_camera_model(stack, offset=100.0)
    assert from_list == from_array
    single = estimate_camera_model(stack[0], offset=100.0)
    assert single.gain == pytest.approx(2.0, rel=0.10)


def test_rejects_too_small_input():
    with pytest.raises(ValueError):
        estimate_camera_model(np.zeros((10, 10)))


def test_unit_conversions_and_dict_roundtrip():
    camera = CameraModel(offset=100.0, gain=2.0, read_std=1.0)
    adu = np.array([90.0, 100.0, 250.5])
    np.testing.assert_allclose(camera.to_adu(camera.to_electrons(adu)), adu)
    assert camera.read_var_e == pytest.approx(1.0 + 1.0 / 48.0)
    assert CameraModel.from_dict(camera.to_dict()) == camera
