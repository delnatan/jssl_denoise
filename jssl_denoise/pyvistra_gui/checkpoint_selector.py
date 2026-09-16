"""Reusable checkpoint-source widget for the jssl_denoise plugin dialog.

Not a reuse of pyvistra's ``SourceSelector`` -- that widget picks an open
*image window* or a file, but a checkpoint is neither: it's a ``.pt`` file
on disk, or (right after a Train run) an in-memory dict this session
already holds. This widget covers both, so the Denoise/Batch tabs can ask
for a ready-to-use denoiser without caring which -- nor which algorithm
trained it: the checkpoint's ``method`` key picks `Denoiser` (Ollion
D-Net/N-Net) or `PoissonDenoiser` (Bayesian Poisson-Gaussian).
"""

from __future__ import annotations

from typing import Optional

import torch
from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from jssl_denoise.checkpoint import load_checkpoint
from jssl_denoise.inference import Denoiser
from jssl_denoise.poisson import PoissonDenoiser
from jssl_denoise.poisson.training import CHECKPOINT_METHOD as POISSON_METHOD


def is_poisson_checkpoint(checkpoint: Optional[dict]) -> bool:
    return (checkpoint or {}).get("method") == POISSON_METHOD


def describe_checkpoint(checkpoint: dict) -> str:
    cfg = checkpoint.get("config", {})
    shape = f"base_filters={cfg.get('base_filters', '?')}  tile_size={cfg.get('tile_size', '?')}"
    if not is_poisson_checkpoint(checkpoint):
        return f"JSSL (D-Net/N-Net)  {shape}"
    cam = checkpoint.get("camera", {})
    return (
        f"Bayesian Poisson-Gaussian  {shape}\n"
        f"camera: offset={cam.get('offset', float('nan')):.2f} ADU  "
        f"gain={cam.get('gain', float('nan')):.3f} ADU/e-  "
        f"read noise={cam.get('read_std', float('nan')):.3f} e-"
    )


class CheckpointSelector(QGroupBox):
    """Group box exposing a session-trained-checkpoint-or-file input.

    Signals
    -------
    checkpoint_changed
        Emitted whenever the selection changes, or a new session checkpoint
        becomes available.
    """

    checkpoint_changed = Signal()

    _SESSION = "__session__"
    _LOAD_FILE = "__file__"

    def __init__(self, parent=None, *, title: str = "Checkpoint"):
        super().__init__(title, parent)

        self._session_checkpoint: Optional[dict] = None
        self._session_summary = ""
        self._file_path: Optional[str] = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        row = QHBoxLayout()
        self.combo = QComboBox()
        self.combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.combo.currentIndexChanged.connect(self._on_combo_changed)
        row.addWidget(self.combo, 1)
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self._browse_file)
        row.addWidget(self.browse_btn)
        outer.addLayout(row)

        self.info_label = QLabel("No checkpoint selected.")
        self.info_label.setStyleSheet("color: #888; font-size: 10px;")
        self.info_label.setWordWrap(True)
        outer.addWidget(self.info_label)

        self._refresh_combo()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_session_checkpoint(self, checkpoint: dict, summary: str) -> None:
        """Register (or replace) the checkpoint a Train run just produced,
        and select it -- called by the Train tab on `finished`/`cancelled`.
        """
        self._session_checkpoint = checkpoint
        self._session_summary = summary
        self._refresh_combo()
        idx = self.combo.findData(self._SESSION)
        if idx >= 0:
            self.combo.setCurrentIndex(idx)

    def current_checkpoint_dict(self) -> Optional[dict]:
        """Return the raw checkpoint dict for the current selection, loading
        it from disk first if a file is selected. `None` if nothing usable
        is selected.
        """
        data = self.combo.currentData()
        if data == self._SESSION:
            return self._session_checkpoint
        if data == self._LOAD_FILE and self._file_path:
            return load_checkpoint(self._file_path)
        return None

    def build_denoiser(
        self, device: torch.device
    ) -> Optional[Denoiser | PoissonDenoiser]:
        """Build a ready-to-use denoiser for the current selection, of the
        class matching the checkpoint's algorithm, or `None` if nothing
        usable is selected. Both classes share `denoise(image, tta)`."""
        ckpt = self.current_checkpoint_dict()
        if ckpt is None:
            return None
        if is_poisson_checkpoint(ckpt):
            return PoissonDenoiser.from_checkpoint(ckpt, device)
        return Denoiser.from_checkpoint(ckpt, device)

    def current_is_poisson(self) -> bool:
        try:
            return is_poisson_checkpoint(self.current_checkpoint_dict())
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _refresh_combo(self) -> None:
        previous = self.combo.currentData()

        self.combo.blockSignals(True)
        self.combo.clear()
        if self._session_checkpoint is not None:
            self.combo.addItem(
                f"(session) {self._session_summary}", self._SESSION
            )
        if self._file_path:
            if self._session_checkpoint is not None:
                self.combo.insertSeparator(self.combo.count())
            self.combo.addItem(
                f"File: {self._file_path.rsplit('/', 1)[-1]}", self._LOAD_FILE
            )

        idx = self.combo.findData(previous) if previous is not None else -1
        if idx < 0:
            idx = 0
        if self.combo.count():
            self.combo.setCurrentIndex(idx)

        self.combo.blockSignals(False)
        self._update_info()

    def _on_combo_changed(self, _index: int) -> None:
        self._update_info()
        self.checkpoint_changed.emit()

    def _update_info(self) -> None:
        data = self.combo.currentData()
        if data == self._SESSION and self._session_checkpoint is not None:
            self.info_label.setText(
                describe_checkpoint(self._session_checkpoint)
            )
        elif data == self._LOAD_FILE and self._file_path:
            try:
                self.info_label.setText(
                    describe_checkpoint(load_checkpoint(self._file_path))
                )
            except Exception as exc:
                self.info_label.setText(f"Failed to read checkpoint: {exc}")
        else:
            self.info_label.setText("No checkpoint selected.")

    def _browse_file(self) -> None:
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load Checkpoint", filter="Checkpoint (*.pt);;All files (*)"
        )
        if not filepath:
            return
        self._file_path = filepath
        self._refresh_combo()
        idx = self.combo.findData(self._LOAD_FILE)
        if idx >= 0:
            self.combo.setCurrentIndex(idx)
