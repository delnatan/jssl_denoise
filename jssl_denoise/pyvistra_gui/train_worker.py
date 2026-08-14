"""Training worker -- pyvistra plugin for jssl_denoise.

Wraps `jssl_denoise.training.Trainer.fit`, run on its own QThread by
`denoise_dialog.DenoiseDialog._start_training`.
"""

from __future__ import annotations

from qtpy.QtCore import QObject, Signal

from jssl_denoise.training import Trainer, _select_device

from ._device_cleanup import release_device_cache


class TrainingWorker(QObject):
    """Train a D-Net/N-Net pair on `images`, reporting progress via Qt
    signals. The worker itself duck-types `jssl_denoise.callbacks.
    TrainingCallback` (a structural Protocol -- `on_step_end`/`on_epoch_end`,
    plus the optional `on_epoch_metrics` hook `Trainer.fit` also checks for),
    so it's passed to `Trainer.fit` directly as `callback=self`: no separate
    adapter class translating callback calls into signal emissions.
    """

    step_progress = Signal(
        int, int, int, float
    )  # step, total_steps, epoch, loss
    epoch_finished = Signal(
        int, int, float, float
    )  # epoch, total_epochs, loss, lr
    epoch_metrics = Signal(
        int, dict
    )  # epoch, {"mu_mse": ..., "sigma_mean": ...}
    finished = Signal(object)  # checkpoint dict
    cancelled = Signal(object)  # checkpoint dict -- see run()
    error = Signal(str)

    def __init__(self, images, config):
        super().__init__()
        self._images = images
        self._config = config
        self._cancel_requested = False

    def cancel(self):
        self._cancel_requested = True

    # ------------------------------------------------------------------
    # TrainingCallback protocol
    # ------------------------------------------------------------------

    def on_step_end(self, step, total_steps, epoch, loss):
        self.step_progress.emit(step, total_steps, epoch, loss)

    def on_epoch_end(self, epoch, total_epochs, loss, lr):
        self.epoch_finished.emit(epoch, total_epochs, loss, lr)

    def on_epoch_metrics(self, epoch, metrics):
        self.epoch_metrics.emit(epoch, dict(metrics))

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self):
        device = _select_device(self._config.device)
        try:
            checkpoint = Trainer(self._config).fit(
                self._images,
                callback=self,
                should_stop=lambda: self._cancel_requested,
            )
        except Exception as exc:
            self.error.emit(str(exc))
            return
        finally:
            # Training allocates/frees activation tensors on every step,
            # thousands of times per run -- release MPS/CUDA's cache so a
            # long-lived GUI session doesn't accumulate pinned device
            # memory across repeated Train runs (see _device_cleanup.py).
            release_device_cache(device)

        # Trainer.fit's should_stop early-return path builds the checkpoint
        # from current weights exactly like a normal finish does (see
        # Trainer._checkpoint / the `if should_stop(): return self.
        # _checkpoint(...)` branch in training.py) -- so a cancelled run
        # still hands back a usable checkpoint rather than nothing.
        if self._cancel_requested:
            self.cancelled.emit(checkpoint)
        else:
            self.finished.emit(checkpoint)
