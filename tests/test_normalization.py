import numpy as np

from jssl_denoise.normalization import RobustNormalizer


def test_invertible_roundtrip():
    rng = np.random.default_rng(0)
    image = rng.normal(loc=1000, scale=50, size=(64, 64)).astype(np.uint16)

    normalizer = RobustNormalizer.fit(image)
    normalized = normalizer.transform(image)
    restored = normalizer.inverse_transform(normalized)

    np.testing.assert_allclose(restored, image.astype(np.float32), atol=1e-2)


def test_fit_pools_across_stack():
    rng = np.random.default_rng(1)
    stack = rng.poisson(lam=100, size=(5, 32, 32)).astype(np.uint16)

    normalizer = RobustNormalizer.fit(stack)

    assert normalizer.scale > 0
    assert 50 < normalizer.mode < 150


def test_fit_accepts_list_of_frames():
    rng = np.random.default_rng(2)
    frames = [
        rng.poisson(lam=100, size=(20, 30)).astype(np.uint16) for _ in range(4)
    ]

    normalizer = RobustNormalizer.fit(frames)

    assert normalizer.scale > 0
