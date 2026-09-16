import numpy as np
import pytest

from jssl_denoise.checkpoint import save_checkpoint
from jssl_denoise.poisson import (
    PoissonDenoiser,
    PoissonTrainer,
    PoissonTrainingConfig,
)


def _tiny_config(**overrides) -> PoissonTrainingConfig:
    cfg = PoissonTrainingConfig(
        epochs=3,
        steps_per_epoch=5,
        tile_size=32,
        tiles_per_batch=8,
        base_filters=8,
        device="cpu",
        seed=0,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


@pytest.fixture
def stack(poisson_gaussian_stack):
    return poisson_gaussian_stack(
        np.random.default_rng(0), n_frames=4, size=128
    )


def test_fit_returns_loadable_checkpoint(tmp_path, stack):
    ckpt = PoissonTrainer(_tiny_config(offset=100.0)).fit(stack)

    assert ckpt["method"] == "poisson"
    assert ckpt["camera"]["offset"] == 100.0
    assert ckpt["camera"]["gain"] > 0
    assert "prior_net_state_dict" in ckpt

    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, ckpt)
    denoised, posterior_std = PoissonDenoiser.load(path, device="cpu").denoise(
        stack[0]
    )
    assert denoised.shape == stack[0].shape
    assert np.all(np.isfinite(denoised))
    assert np.all(posterior_std > 0)


def test_should_stop_halts_training_early(stack):
    calls = {"n": 0}

    def should_stop():
        calls["n"] += 1
        return calls["n"] > 2

    ckpt = PoissonTrainer(_tiny_config(epochs=10, steps_per_epoch=10)).fit(
        stack, should_stop=should_stop
    )
    assert ckpt["method"] == "poisson"
    assert calls["n"] <= 10


def test_loss_decreases_with_more_training(stack):
    losses = []

    class RecordingCallback:
        def on_step_end(self, step, total_steps, epoch, loss):
            pass

        def on_epoch_end(self, epoch, total_epochs, loss, lr):
            losses.append(loss)

    cfg = _tiny_config(epochs=20, steps_per_epoch=10, seed=1)
    PoissonTrainer(cfg).fit(stack, callback=RecordingCallback())

    assert np.mean(losses[-5:]) < np.mean(losses[:5])


def test_early_stopping_reports_best_epoch_loss(stack):
    losses, early_stops = [], []

    class RecordingCallback:
        def on_step_end(self, step, total_steps, epoch, loss):
            pass

        def on_epoch_end(self, epoch, total_epochs, loss, lr):
            losses.append(loss)

        def on_early_stop(self, epoch, best_epoch, best_metric):
            early_stops.append((epoch, best_epoch, best_metric))

    cfg = _tiny_config(
        epochs=25, steps_per_epoch=5, early_stop_patience=1, seed=2
    )
    PoissonTrainer(cfg).fit(stack, callback=RecordingCallback())

    assert len(losses) < cfg.epochs
    assert len(early_stops) == 1
    stopped, best_epoch, best_loss = early_stops[0]
    assert stopped == len(losses)
    assert best_epoch <= stopped
    assert best_loss == min(losses)
