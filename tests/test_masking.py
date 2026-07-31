import numpy as np
import torch

from jssl_denoise.masking import (
    NEIGHBOR_KERNEL,
    apply_masking,
    gaussian_neighbor_average,
    make_grid_mask,
    sample_grid_spacing,
)


def test_mask_density_near_paper_value():
    rng = np.random.default_rng(0)
    fractions = []
    for _ in range(200):
        spacing = sample_grid_spacing(rng)
        mask = make_grid_mask((256, 256), spacing, rng)
        fractions.append(mask.mean())

    mean_fraction = float(np.mean(fractions))
    assert 0.05 < mean_fraction < 0.09


def test_neighbor_kernel_sums_to_one_and_ignores_center():
    assert torch.isclose(NEIGHBOR_KERNEL.sum(), torch.tensor(1.0), atol=1e-6)
    assert NEIGHBOR_KERNEL[0, 0, 1, 1] == 0.0


def test_gaussian_neighbor_average_ignores_center_pixel():
    # a unit impulse at the center: since the kernel's center weight is exactly
    # 0, the output at the impulse's own location must be unaffected by it,
    # while its 8 neighbors' outputs must change.
    image = torch.zeros(1, 1, 16, 16)
    image[0, 0, 8, 8] = 1.0

    avg = gaussian_neighbor_average(image)

    assert torch.isclose(avg[0, 0, 8, 8], torch.tensor(0.0), atol=1e-6)
    assert avg[0, 0, 7, 8] > 0
    assert avg[0, 0, 9, 9] > 0


def test_apply_masking_replaces_only_masked_pixels():
    rng = np.random.default_rng(0)
    image = torch.from_numpy(rng.normal(size=(2, 1, 16, 16)).astype(np.float32))
    mask = torch.from_numpy(make_grid_mask((16, 16), 4, rng))

    masked_image, g_values = apply_masking(image, mask)

    mask_4d = mask.view(1, 1, 16, 16).expand_as(image)
    assert torch.equal(masked_image[~mask_4d], image[~mask_4d])
    assert torch.equal(masked_image[mask_4d], g_values[mask_4d])
