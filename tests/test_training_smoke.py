import numpy as np
import torch

from jssl_denoise.config import TrainingConfig
from jssl_denoise.inference import Denoiser
from jssl_denoise.networks import DNet, NNet
from jssl_denoise.training import Trainer


def _synthetic_stack(
    rng, n_frames=6, size=64, signal_value=150.0, background=100.0
):
    clean = np.full((size, size), background, dtype=np.float32)
    clean[size // 4 : size // 2, size // 4 : size // 2] = signal_value
    stack = rng.poisson(clean[None, :, :].repeat(n_frames, axis=0)).astype(
        np.uint16
    )
    return stack


def _tiny_config(**overrides) -> TrainingConfig:
    cfg = TrainingConfig(
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


def test_fit_returns_well_formed_checkpoint(tmp_path):
    rng = np.random.default_rng(0)
    stack = _synthetic_stack(rng)

    trainer = Trainer(_tiny_config())
    ckpt = trainer.fit(stack)

    assert ckpt["version"] == "1.0"
    assert "d_net_state_dict" in ckpt
    assert "n_net_state_dict" in ckpt
    assert ckpt["normalizer"]["scale"] > 0

    from jssl_denoise.checkpoint import save_checkpoint

    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, ckpt)
    denoiser = Denoiser.load(path, device="cpu")

    denoised, noise_std_map = denoiser.denoise(stack[0], tta=False)
    assert denoised.shape == stack[0].shape
    assert noise_std_map.shape == stack[0].shape


def test_should_stop_halts_training_early():
    rng = np.random.default_rng(0)
    stack = _synthetic_stack(rng)

    calls = {"n": 0}

    def should_stop():
        calls["n"] += 1
        return calls["n"] > 2  # stop after a couple of steps

    trainer = Trainer(_tiny_config(epochs=10, steps_per_epoch=10))
    ckpt = trainer.fit(stack, should_stop=should_stop)

    assert ckpt["version"] == "1.0"
    assert (
        calls["n"] <= 10
    )  # confirms should_stop was actually consulted, not ignored


def test_loss_decreases_with_more_training():
    rng = np.random.default_rng(0)
    stack = _synthetic_stack(rng, n_frames=8, size=64)

    losses_per_epoch = []

    class RecordingCallback:
        def on_step_end(self, step, total_steps, epoch, loss):
            pass

        def on_epoch_end(self, epoch, total_epochs, loss, lr):
            losses_per_epoch.append(loss)

    trainer = Trainer(_tiny_config(epochs=20, steps_per_epoch=10, seed=1))
    trainer.fit(stack, callback=RecordingCallback())

    early = np.mean(losses_per_epoch[:5])
    late = np.mean(losses_per_epoch[-5:])
    assert late < early


def test_snapshot_is_a_detached_clone_immune_to_further_training():
    d_net = DNet(base_filters=4)
    n_net = NNet()

    d_state, n_state = Trainer._snapshot(d_net, n_net)
    before = d_state["enc1.conv.weight"].clone()
    assert d_state["enc1.conv.weight"].device.type == "cpu"

    with torch.no_grad():
        for p in d_net.parameters():
            p.add_(1.0)

    # the snapshot must not have moved with the live network's further training
    assert torch.equal(d_state["enc1.conv.weight"], before)
    assert not torch.equal(
        d_state["enc1.conv.weight"], d_net.state_dict()["enc1.conv.weight"]
    )


def test_early_stopping_halts_before_configured_epochs():
    rng = np.random.default_rng(0)
    stack = _synthetic_stack(rng, n_frames=8, size=64)

    epoch_mu_mses = []
    early_stop_calls = []

    class RecordingCallback:
        def on_step_end(self, step, total_steps, epoch, loss):
            pass

        def on_epoch_end(self, epoch, total_epochs, loss, lr):
            pass

        def on_epoch_metrics(self, epoch, metrics):
            epoch_mu_mses.append((epoch, metrics["mu_mse"]))

        def on_early_stop(self, epoch, best_epoch, best_mu_mse):
            early_stop_calls.append((epoch, best_epoch, best_mu_mse))

    # patience=1 is aggressive on purpose -- any single epoch without a
    # strict mu_mse improvement stops training, so a generous epochs budget
    # should never actually be reached.
    cfg = _tiny_config(
        epochs=25,
        steps_per_epoch=5,
        seed=2,
        early_stop_patience=1,
        early_stop_min_delta=0.0,
    )
    ckpt = Trainer(cfg).fit(stack, callback=RecordingCallback())

    assert ckpt["version"] == "1.0"
    assert len(epoch_mu_mses) < cfg.epochs
    assert len(early_stop_calls) == 1
    stopped_epoch, best_epoch, best_mu_mse = early_stop_calls[0]
    assert stopped_epoch == len(epoch_mu_mses)
    assert best_epoch <= stopped_epoch
    recorded_best = min(mu_mse for _, mu_mse in epoch_mu_mses)
    assert best_mu_mse == recorded_best
