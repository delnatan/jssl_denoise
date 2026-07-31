from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class TrainingCallback(Protocol):
    """Plain-Python progress hook for `Trainer.fit`. Deliberately Qt-agnostic
    (no signals/threads) so a GUI (e.g. a PyQt worker's `run()`) can wrap an
    instance of this protocol around its own Signals without this package
    ever importing Qt.
    """

    def on_step_end(self, step: int, total_steps: int, epoch: int, loss: float) -> None: ...

    def on_epoch_end(self, epoch: int, total_epochs: int, loss: float, lr: float) -> None: ...

    # `Trainer` also duck-types an *optional* `on_epoch_metrics(self, epoch,
    # metrics: dict[str, float])` method for extra per-epoch diagnostics
    # (e.g. mu's MSE against the raw target, mean sigma) that aren't part of
    # the loss itself -- deliberately left off this Protocol (and so out of
    # its runtime_checkable isinstance() contract) so existing callbacks that
    # only implement the two methods above keep satisfying it unchanged.
    # See `ConsoleCallback` for a callback that implements it.


class ConsoleCallback:
    """Default callback: an in-place, flushed step counter (so progress is
    visible immediately even when stdout is piped/redirected, which
    block-buffers by default), plus a one-line summary per epoch.
    """

    def __init__(self, step_every: int = 1):
        self.step_every = step_every

    def on_step_end(self, step: int, total_steps: int, epoch: int, loss: float) -> None:
        if step % self.step_every == 0 or step == total_steps:
            print(
                f"\repoch {epoch} step {step}/{total_steps}  loss={loss:.4f}",
                end="",
                flush=True,
            )

    def on_epoch_end(self, epoch: int, total_epochs: int, loss: float, lr: float) -> None:
        print(f"\repoch {epoch}/{total_epochs}  loss={loss:.4f}  lr={lr:.2e}" + " " * 20, flush=True)

    def on_epoch_metrics(self, epoch: int, metrics: dict[str, float]) -> None:
        parts = "  ".join(f"{name}={value:.4f}" for name, value in metrics.items())
        print(f"          {parts}", flush=True)
