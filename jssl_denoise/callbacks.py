from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class TrainingCallback(Protocol):
    """Plain-Python progress hook for `Trainer.fit`. Deliberately Qt-agnostic
    (no signals/threads) so a GUI (e.g. a PyQt worker's `run()`) can wrap an
    instance of this protocol around its own Signals without this package
    ever importing Qt.
    """

    def on_step_end(
        self, step: int, total_steps: int, epoch: int, loss: float
    ) -> None: ...

    def on_epoch_end(
        self, epoch: int, total_epochs: int, loss: float, lr: float
    ) -> None: ...

    # `Trainer` also duck-types two *optional* methods, deliberately left off
    # this Protocol (and so out of its runtime_checkable isinstance() contract)
    # so existing callbacks that only implement the two methods above keep
    # satisfying it unchanged. See `ConsoleCallback` for a callback that
    # implements both:
    #
    #   on_epoch_metrics(self, epoch, metrics: dict[str, float]) -- extra
    #   per-epoch diagnostics (mu's MSE against the raw target, mean sigma)
    #   that aren't part of the loss itself.
    #
    #   on_early_stop(self, epoch, best_epoch, best_mu_mse) -- fired once, in
    #   place of the epoch loop's normal exit, when `early_stop_patience`
    #   epochs pass with no mu_mse improvement (see config.TrainingConfig).


class ConsoleCallback:
    """Default callback: an in-place, flushed step counter (so progress is
    visible immediately even when stdout is piped/redirected, which
    block-buffers by default), plus a one-line summary per epoch.
    """

    def __init__(self, step_every: int = 1):
        self.step_every = step_every

    def on_step_end(
        self, step: int, total_steps: int, epoch: int, loss: float
    ) -> None:
        if step % self.step_every == 0 or step == total_steps:
            print(
                f"\repoch {epoch} step {step}/{total_steps}  loss={loss:.4f}",
                end="",
                flush=True,
            )

    def on_epoch_end(
        self, epoch: int, total_epochs: int, loss: float, lr: float
    ) -> None:
        print(
            f"\repoch {epoch}/{total_epochs}  loss={loss:.4f}  lr={lr:.2e}"
            + " " * 20,
            flush=True,
        )

    def on_epoch_metrics(self, epoch: int, metrics: dict[str, float]) -> None:
        parts = "  ".join(
            f"{name}={value:.4f}" for name, value in metrics.items()
        )
        print(f"          {parts}", flush=True)

    def on_early_stop(
        self, epoch: int, best_epoch: int, best_mu_mse: float
    ) -> None:
        print(
            f"early stopping at epoch {epoch} -- no mu_mse improvement since "
            f"epoch {best_epoch} (best mu_mse={best_mu_mse:.4f}); returning "
            f"that epoch's weights",
            flush=True,
        )
