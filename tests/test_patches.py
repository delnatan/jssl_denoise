import numpy as np

from jssl_denoise.patches import FrameSampler, sample_patch_batch


def test_frame_sampler_covers_every_frame_before_repeating():
    rng = np.random.default_rng(0)
    sampler = FrameSampler(n_frames=5, rng=rng)

    first_pass = sorted(sampler.next() for _ in range(5))
    assert first_pass == [0, 1, 2, 3, 4]

    second_pass = sorted(sampler.next() for _ in range(5))
    assert second_pass == [0, 1, 2, 3, 4]


def test_frame_sampler_reshuffles_across_many_passes():
    # over enough passes the draw order shouldn't be identical every time --
    # a stuck/non-reshuffling implementation would fail this.
    rng = np.random.default_rng(0)
    sampler = FrameSampler(n_frames=8, rng=rng)

    passes = [[sampler.next() for _ in range(8)] for _ in range(10)]
    assert len({tuple(p) for p in passes}) > 1


def test_sample_patch_batch_uses_every_frame_at_least_once_per_full_pass():
    rng = np.random.default_rng(0)
    frames = [np.full((32, 32), fill_value=i, dtype=np.float32) for i in range(4)]
    sampler = FrameSampler(n_frames=len(frames), rng=rng)

    batch = sample_patch_batch(
        frames, tile_size=8, n_tiles=len(frames), rng=rng, sampler=sampler, augment=False
    )

    seen_values = {float(batch[i].min()) for i in range(len(frames))}
    assert seen_values == {0.0, 1.0, 2.0, 3.0}
