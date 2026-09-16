"""Denoise dialog -- pyvistra plugin for jssl_denoise.

Requires ``jssl_denoise[gui]`` (pulls in ``pyvistra``, ``qtpy``). Registered
into pyvistra's ``Image -> Denoising`` menu by
``jssl_denoise/_pyvistra_plugin.py``.

Four tabs: Train (fits a new model with the selected algorithm -- JSSL
D-Net/N-Net via ``Trainer.fit``, or the Bayesian Poisson-Gaussian prior via
``poisson.PoissonTrainer.fit``), Denoise (runs a trained checkpoint over one
source; the checkpoint itself says which algorithm it is), Batch (the
Denoise tab's settings replayed across every checked file in a folder), and
Monitor (shared loss/metric plot + run log, reused across all three).

Layout/threading conventions (source/region selection, output routing,
per-T-frame QThread chaining, Force-Stop escape hatch) mirror resolvde's
``pyvistra_gui/decon_dialog.py`` -- read in full before writing this file.
The one structural addition decon doesn't need is the Train tab, since
jssl_denoise requires fitting a model before any inference is possible.
"""

from __future__ import annotations

import os
from datetime import datetime

import numpy as np
from qtpy.QtCore import Qt, QThread, QTimer
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from jssl_denoise.checkpoint import save_checkpoint
from jssl_denoise.config import TrainingConfig
from jssl_denoise.poisson import PoissonTrainer, PoissonTrainingConfig
from jssl_denoise.training import Trainer, _select_device

from .batch_worker import BatchDenoiseRunner
from .checkpoint_selector import CheckpointSelector, is_poisson_checkpoint
from .denoise_worker import DenoiseWorker
from .train_worker import TrainingWorker


ALGORITHM_JSSL = "jssl"
ALGORITHM_POISSON = "poisson"

# Right-axis metric each trainer reports via on_epoch_metrics.
_MONITOR_METRIC = {ALGORITHM_JSSL: "mu_mse", ALGORITHM_POISSON: "prior_mse_e"}


def _dspin(value, minimum, maximum, step=0.01, decimals=4):
    box = QDoubleSpinBox()
    box.setDecimals(decimals)
    box.setRange(minimum, maximum)
    box.setSingleStep(step)
    box.setValue(value)
    box.setMaximumWidth(104 if decimals >= 6 else 84 if decimals >= 3 else 72)
    return box


def _ispin(value, minimum, maximum):
    box = QSpinBox()
    box.setRange(minimum, maximum)
    box.setValue(value)
    box.setMaximumWidth(80)
    return box


def _tight_form(form: QFormLayout) -> QFormLayout:
    form.setLabelAlignment(Qt.AlignRight)
    form.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)
    form.setHorizontalSpacing(10)
    form.setVerticalSpacing(8)
    return form


def _z_stack_warning_label() -> QLabel:
    """Hidden by default; shown when the selected source has Z > 1.

    jssl_denoise only ever sees one Z plane at a time (set via the Z-plane
    spinbox) -- it's a 2D/2D+time method, not a volumetric one. pyvistra
    already normalizes every source to (T, Z, C, Y, X), so Z from that
    shape is a reliable signal that a file is a genuine z-stack rather
    than a single plane, and worth flagging before the other Z planes are
    silently skipped.
    """
    label = QLabel()
    label.setWordWrap(True)
    label.setStyleSheet("color: #e0a030;")
    label.setVisible(False)
    return label


def _update_z_stack_warning(label: QLabel, Z: int) -> None:
    if Z > 1:
        label.setText(
            f"⚠ Source has {Z} Z-planes. jssl_denoise is a 2D/2D+time method -- "
            "only the Z-plane selected above is used; the rest are ignored."
        )
        label.setVisible(True)
    else:
        label.setVisible(False)


class _Advanced(QGroupBox):
    """Checkable group box that hides its form body until expanded."""

    def __init__(self, title="Advanced"):
        super().__init__(title)
        self.setCheckable(True)
        self.setChecked(False)
        body = QWidget()
        self.form = _tight_form(QFormLayout(body))
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.addWidget(body)
        body.setVisible(False)
        self.toggled.connect(body.setVisible)


def _device_combo() -> QComboBox:
    import torch

    combo = QComboBox()
    combo.addItem("Auto", None)
    combo.addItem("CPU", "cpu")
    cuda_idx = combo.count()
    combo.addItem("CUDA", "cuda")
    if not torch.cuda.is_available():
        combo.model().item(cuda_idx).setEnabled(False)
    mps_idx = combo.count()
    combo.addItem("MPS", "mps")
    if not torch.backends.mps.is_available():
        combo.model().item(mps_idx).setEnabled(False)
    return combo


class DenoiseDialog(QDialog):
    def __init__(self, viewer=None, parent=None):
        super().__init__(parent)
        from pyvistra.widgets import (
            BufferProcessingRunner,
            ConvergencePlotWidget,
            FlaggableFileListWidget,
            ImageOutputSelector,
            RegionSelector,
            SourceSelector,
        )

        self.viewer = viewer
        self.setWindowTitle("Denoise")
        self.setWindowFlags(Qt.Tool)
        self.resize(480, 760)
        self._batch_folder = None
        self._train_thread = None
        self._train_worker = None
        self._last_epoch_info = {}

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(8, 8, 8, 8)
        outer_layout.setSpacing(8)

        self.tabs = QTabWidget()
        outer_layout.addWidget(self.tabs, 1)

        def _scroll_tab(title):
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.NoFrame)
            content = QWidget()
            layout = QVBoxLayout(content)
            layout.setContentsMargins(4, 4, 4, 4)
            layout.setSpacing(10)
            scroll.setWidget(content)
            self.tabs.addTab(scroll, title)
            return layout

        train_layout = _scroll_tab("Train")
        denoise_layout = _scroll_tab("Denoise")

        self._build_train_tab(
            train_layout, SourceSelector, RegionSelector, viewer
        )
        self._build_denoise_tab(
            denoise_layout,
            SourceSelector,
            RegionSelector,
            ImageOutputSelector,
            BufferProcessingRunner,
            viewer,
        )
        self._build_batch_tab(FlaggableFileListWidget)
        self._build_monitor_tab(ConvergencePlotWidget)

        self._on_train_source_changed()
        self._on_source_changed()
        self._on_checkpoint_changed()

    # ------------------------------------------------------------------
    # Train tab
    # ------------------------------------------------------------------

    def _build_train_tab(self, layout, SourceSelector, RegionSelector, viewer):
        algo_box = QGroupBox("Algorithm")
        algo_layout = QVBoxLayout(algo_box)
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItem(
            "JSSL -- D-Net/N-Net (Ollion et al. 2021)", ALGORITHM_JSSL
        )
        self.algorithm_combo.addItem(
            "Bayesian Poisson-Gaussian (de Wolf et al. 2026)",
            ALGORITHM_POISSON,
        )
        algo_layout.addWidget(self.algorithm_combo)
        self.algorithm_info_label = QLabel()
        self.algorithm_info_label.setWordWrap(True)
        self.algorithm_info_label.setStyleSheet("color: #888; font-size: 10px;")
        algo_layout.addWidget(self.algorithm_info_label)
        layout.addWidget(algo_box)

        self.train_source_selector = SourceSelector(title="Source")
        if viewer is not None:
            self.train_source_selector.set_default_window(viewer)
        self.train_source_selector.source_changed.connect(
            self._on_train_source_changed
        )
        layout.addWidget(self.train_source_selector)

        self.train_region_selector = RegionSelector(title="Region (optional)")
        layout.addWidget(self.train_region_selector)

        tc_box = QGroupBox("Frame / Plane / Channel")
        tc_form = _tight_form(QFormLayout(tc_box))
        frame_row = QHBoxLayout()
        self.train_t_start_spin = _ispin(0, 0, 0)
        self.train_t_end_spin = _ispin(0, 0, 0)
        frame_row.addWidget(QLabel("T:"))
        frame_row.addWidget(self.train_t_start_spin)
        frame_row.addWidget(QLabel("to"))
        frame_row.addWidget(self.train_t_end_spin)
        frame_row.addStretch(1)
        tc_form.addRow("Frame range:", frame_row)
        self.train_z_spin = _ispin(0, 0, 0)
        tc_form.addRow("Z plane:", self.train_z_spin)
        self.train_channel_spin = _ispin(0, 0, 0)
        self.train_channel_spin.setToolTip(
            "D-Net is single-channel -- one checkpoint denoises one channel. "
            "Train a separate checkpoint per channel if needed."
        )
        tc_form.addRow("Channel:", self.train_channel_spin)
        self.train_z_warning_label = _z_stack_warning_label()
        tc_form.addRow(self.train_z_warning_label)
        layout.addWidget(tc_box)

        cfg_box = QGroupBox("Training")
        cfg_form = _tight_form(QFormLayout(cfg_box))
        self.epochs_spin = _ispin(60, 1, 5000)
        self.steps_per_epoch_spin = _ispin(100, 1, 5000)
        self.tile_size_spin = _ispin(64, 16, 1024)
        self.tiles_per_batch_spin = _ispin(32, 1, 512)
        self.lr_spin = _dspin(4e-4, 1e-6, 1.0, 1e-4, decimals=6)
        self.base_filters_spin = _ispin(64, 4, 256)
        self.nll_beta_spin = _dspin(0.5, 0.0, 1.0, 0.05, decimals=3)
        self.nll_beta_spin.setToolTip(
            "beta-NLL reweighting: 0 is the exact NLL (N-Net's sigma can\n"
            "decouple from D-Net's mu actually improving); 1 makes mu's\n"
            "gradient behave like plain MSE. 0.5 is a middle ground."
        )
        self.augment_cb = QCheckBox("Augment (flips/rotations)")
        self.augment_cb.setChecked(True)
        self.train_device_combo = _device_combo()
        cfg_form.addRow("Epochs:", self.epochs_spin)
        cfg_form.addRow("Steps/epoch:", self.steps_per_epoch_spin)
        cfg_form.addRow("Tile size:", self.tile_size_spin)
        cfg_form.addRow("Tiles/batch:", self.tiles_per_batch_spin)
        cfg_form.addRow("Learning rate:", self.lr_spin)
        cfg_form.addRow("Base filters:", self.base_filters_spin)
        self.nll_beta_label = QLabel("NLL beta:")
        cfg_form.addRow(self.nll_beta_label, self.nll_beta_spin)
        cfg_form.addRow(self.augment_cb)
        cfg_form.addRow("Device:", self.train_device_combo)
        layout.addWidget(cfg_box)

        self.camera_box = QGroupBox("Camera")
        camera_form = _tight_form(QFormLayout(self.camera_box))
        camera_note = QLabel(
            "Gain and read noise are fitted from the raw images (no dark "
            "frames), then held fixed. Input must be raw camera ADU. Give "
            "the camera's black level for a physical read-noise estimate; "
            "otherwise it defaults to the darkest regions' level."
        )
        camera_note.setWordWrap(True)
        camera_note.setStyleSheet("color: #888; font-size: 10px;")
        camera_form.addRow(camera_note)
        self.offset_cb = QCheckBox("Known offset (ADU):")
        self.offset_spin = _dspin(100.0, -1e6, 1e6, 1.0, decimals=2)
        self.offset_spin.setEnabled(False)
        self.offset_cb.toggled.connect(self.offset_spin.setEnabled)
        camera_form.addRow(self.offset_cb, self.offset_spin)
        self.gain_cb = QCheckBox("Known gain (ADU/e-):")
        self.gain_spin = _dspin(2.0, 1e-4, 1e4, 0.01, decimals=4)
        self.gain_spin.setEnabled(False)
        self.gain_cb.toggled.connect(self.gain_spin.setEnabled)
        camera_form.addRow(self.gain_cb, self.gain_spin)
        layout.addWidget(self.camera_box)

        adv = _Advanced()
        self.lr_decay_factor_spin = _dspin(0.5, 0.01, 1.0, 0.05, decimals=3)
        self.lr_decay_every_spin = _ispin(30, 1, 1000)
        self.lr_floor_spin = _dspin(1e-6, 0.0, 1.0, 1e-6, decimals=8)
        self.mask_spacing_low_spin = _ispin(3, 1, 50)
        self.mask_spacing_high_spin = _ispin(5, 1, 50)
        self.seed_cb = QCheckBox("Fixed seed")
        self.seed_spin = _ispin(0, 0, 2**31 - 1)
        self.seed_spin.setEnabled(False)
        self.seed_cb.toggled.connect(self.seed_spin.setEnabled)
        for label, w in [
            ("lr_decay_factor:", self.lr_decay_factor_spin),
            ("lr_decay_every_epochs:", self.lr_decay_every_spin),
            ("lr_floor:", self.lr_floor_spin),
            ("mask_spacing_low:", self.mask_spacing_low_spin),
            ("mask_spacing_high:", self.mask_spacing_high_spin),
        ]:
            adv.form.addRow(label, w)
        adv.form.addRow(self.seed_cb)
        adv.form.addRow("Seed:", self.seed_spin)
        adv.setTitle(f"Advanced ({adv.form.rowCount()} params)")
        layout.addWidget(adv)

        ckpt_out_box = QGroupBox("Checkpoint output")
        ckpt_out_form = _tight_form(QFormLayout(ckpt_out_box))
        path_row = QHBoxLayout()
        self.checkpoint_path_edit = QLineEdit()
        self.checkpoint_path_edit.setPlaceholderText(
            "(session only -- not saved to disk)"
        )
        path_row.addWidget(self.checkpoint_path_edit, 1)
        browse_ckpt_btn = QPushButton("Browse...")
        browse_ckpt_btn.clicked.connect(self._browse_checkpoint_save_path)
        path_row.addWidget(browse_ckpt_btn)
        ckpt_out_form.addRow("Save to:", path_row)
        layout.addWidget(ckpt_out_box)
        layout.addStretch(1)

        self.train_progress_bar = QProgressBar()
        layout.addWidget(self.train_progress_bar)
        self.train_status_label = QLabel("Ready")
        self.train_status_label.setWordWrap(True)
        self.train_status_label.setStyleSheet("color: #888;")
        layout.addWidget(self.train_status_label)

        train_buttons = QHBoxLayout()
        train_buttons.addStretch()
        self.train_start_btn = QPushButton("Start Training")
        self.train_start_btn.clicked.connect(self._start_training)
        train_buttons.addWidget(self.train_start_btn)
        self.train_cancel_btn = QPushButton("Cancel")
        self.train_cancel_btn.setEnabled(False)
        self.train_cancel_btn.clicked.connect(self._cancel_training)
        train_buttons.addWidget(self.train_cancel_btn)
        self.train_force_stop_btn = QPushButton("Force Stop")
        self.train_force_stop_btn.setEnabled(False)
        self.train_force_stop_btn.setVisible(False)
        self.train_force_stop_btn.clicked.connect(self._force_stop_training)
        train_buttons.addWidget(self.train_force_stop_btn)
        train_buttons.addStretch()
        layout.addLayout(train_buttons)

        self._train_force_stop_timer = QTimer(self)
        self._train_force_stop_timer.setSingleShot(True)
        self._train_force_stop_timer.setInterval(5000)
        self._train_force_stop_timer.timeout.connect(
            self._reveal_train_force_stop
        )

        self.algorithm_combo.currentIndexChanged.connect(
            self._on_algorithm_changed
        )
        self._on_algorithm_changed()

    def _selected_algorithm(self) -> str:
        return self.algorithm_combo.currentData()

    def _on_algorithm_changed(self, _index=None):
        poisson = self._selected_algorithm() == ALGORITHM_POISSON
        self.nll_beta_label.setVisible(not poisson)
        self.nll_beta_spin.setVisible(not poisson)
        self.camera_box.setVisible(poisson)
        self.algorithm_info_label.setText(
            "Blind-spot network predicts a per-pixel Gamma prior over photon "
            "counts; denoising returns the Bayesian posterior mean, which "
            "folds each pixel's own value back in. Best for low photon "
            "counts and small spots."
            if poisson
            else "Blind-spot D-Net denoiser with an N-Net Poisson+Gaussian "
            "noise-variance model learned jointly (Gaussian likelihood)."
        )

    def _on_train_source_changed(self):
        window = self.train_source_selector.selected_window()
        self.train_region_selector.set_source(window)
        data, _meta = self.train_source_selector.get_source()
        if data is not None:
            T, Z, C, _Y, _X = data.shape
            self.train_t_start_spin.setRange(0, max(0, T - 1))
            self.train_t_end_spin.setRange(0, max(0, T - 1))
            self.train_t_end_spin.setValue(max(0, T - 1))
            self.train_z_spin.setRange(0, max(0, Z - 1))
            self.train_channel_spin.setRange(0, max(0, C - 1))
            _update_z_stack_warning(self.train_z_warning_label, Z)
        else:
            self.train_z_warning_label.setVisible(False)

    def _browse_checkpoint_save_path(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Checkpoint", "checkpoint.pt", "*.pt"
        )
        if path:
            if not path.endswith(".pt"):
                path += ".pt"
            self.checkpoint_path_edit.setText(path)

    def _set_train_status(self, text, error=False, ok=False):
        self.train_status_label.setText(text)
        if error:
            self.train_status_label.setStyleSheet("color: #F44;")
        elif ok:
            self.train_status_label.setStyleSheet("color: #4A4;")
        else:
            self.train_status_label.setStyleSheet("color: #888;")

    def _start_training(self):
        if (
            self._train_thread is not None
            or self._runner.is_running()
            or self._batch_runner.is_running()
        ):
            return

        data, meta = self.train_source_selector.get_source()
        if data is None:
            self._set_train_status("No source selected.", error=True)
            return

        T, Z, C, Y, X = data.shape
        bbox = self.train_region_selector.bbox()
        y0, x0, y1, x1 = bbox if bbox is not None else (0, 0, Y, X)
        t0, t1 = (
            self.train_t_start_spin.value(),
            self.train_t_end_spin.value() + 1,
        )
        if t0 >= t1 or t1 > T:
            self._set_train_status("Invalid T range.", error=True)
            return
        z = self.train_z_spin.value()
        c = self.train_channel_spin.value()

        frames = [
            np.asarray(data[t, z, c, y0:y1, x0:x1]) for t in range(t0, t1)
        ]

        algorithm = self._selected_algorithm()
        common = dict(
            lr=self.lr_spin.value(),
            lr_decay_factor=self.lr_decay_factor_spin.value(),
            lr_decay_every_epochs=self.lr_decay_every_spin.value(),
            lr_floor=self.lr_floor_spin.value(),
            epochs=self.epochs_spin.value(),
            steps_per_epoch=self.steps_per_epoch_spin.value(),
            tile_size=self.tile_size_spin.value(),
            tiles_per_batch=self.tiles_per_batch_spin.value(),
            mask_spacing_low=self.mask_spacing_low_spin.value(),
            mask_spacing_high=self.mask_spacing_high_spin.value(),
            augment=self.augment_cb.isChecked(),
            base_filters=self.base_filters_spin.value(),
            device=self.train_device_combo.currentData(),
            seed=self.seed_spin.value() if self.seed_cb.isChecked() else None,
        )
        if algorithm == ALGORITHM_POISSON:
            config = PoissonTrainingConfig(
                offset=self.offset_spin.value()
                if self.offset_cb.isChecked()
                else None,
                gain=self.gain_spin.value() if self.gain_cb.isChecked() else None,
                **common,
            )
            trainer = PoissonTrainer(config)
        else:
            config = TrainingConfig(nll_beta=self.nll_beta_spin.value(), **common)
            trainer = Trainer(config)
        metric = _MONITOR_METRIC[algorithm]

        if min(y1 - y0, x1 - x0) < config.tile_size:
            self._set_train_status(
                f"Region ({y1 - y0}x{x1 - x0}) is smaller than tile_size ({config.tile_size}).",
                error=True,
            )
            return

        self._last_epoch_info = {"algorithm": algorithm}
        self.train_progress_bar.setRange(
            0, max(1, config.epochs * config.steps_per_epoch)
        )
        self.train_progress_bar.setValue(0)
        self._set_train_status("Training...")
        self.train_start_btn.setEnabled(False)
        self.train_cancel_btn.setEnabled(True)
        self._hide_train_force_stop()
        self._set_monitor_running_indicator(True)
        self.convergence_plot.clear()
        self.convergence_plot.set_labels(
            x_label="Epoch", y_label="loss", y_label_right=metric
        )
        self.log_view.clear()
        self._log(
            f"── train {datetime.now():%Y-%m-%d %H:%M:%S} " + "─" * 20 + "\n"
            f"algorithm={algorithm}\n"
            f"frames={len(frames)}  shape={frames[0].shape}  device={config.device or 'auto'}\n"
            f"epochs={config.epochs}  steps_per_epoch={config.steps_per_epoch}  "
            f"tile_size={config.tile_size}  tiles_per_batch={config.tiles_per_batch}  lr={config.lr}"
        )

        worker = TrainingWorker(frames, trainer)
        self._train_worker = worker
        self._train_thread = QThread()
        worker.moveToThread(self._train_thread)

        self._train_thread.started.connect(worker.run)
        worker.step_progress.connect(self._on_train_step)
        worker.epoch_finished.connect(self._on_train_epoch)
        worker.epoch_metrics.connect(self._on_train_epoch_metrics)
        worker.finished.connect(self._on_train_finished)
        worker.cancelled.connect(self._on_train_cancelled)
        worker.error.connect(self._on_train_error)

        worker.finished.connect(self._train_thread.quit)
        worker.cancelled.connect(self._train_thread.quit)
        worker.error.connect(self._train_thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.cancelled.connect(worker.deleteLater)
        worker.error.connect(worker.deleteLater)
        self._train_thread.finished.connect(self._train_thread.deleteLater)
        self._train_thread.finished.connect(self._on_train_thread_finished)

        self._train_thread.start()

    def _on_train_step(self, step, total_steps, epoch, loss):
        self.train_progress_bar.setValue((epoch - 1) * total_steps + step)
        self._set_train_status(
            f"Training... epoch {epoch}  step {step}/{total_steps}  loss={loss:.4f}"
        )

    def _on_train_epoch(self, epoch, total_epochs, loss, lr):
        self.convergence_plot.append_point("loss", loss, axis="left")
        self._last_epoch_info = {
            "epoch": epoch,
            "total_epochs": total_epochs,
            "loss": loss,
        }
        self._log(f"epoch {epoch}/{total_epochs}  loss={loss:.4f}  lr={lr:.2e}")

    def _on_train_epoch_metrics(self, epoch, metrics):
        metric = _MONITOR_METRIC[self._last_epoch_info.get("algorithm", ALGORITHM_JSSL)]
        value = metrics.get(metric)
        if value is not None:
            self.convergence_plot.append_point(metric, value, axis="right")
            self._last_epoch_info["metric"] = (metric, value)

    def _checkpoint_summary(self) -> str:
        info = getattr(self, "_last_epoch_info", {}) or {}
        parts = [
            f"epoch {info.get('epoch', '?')}/{info.get('total_epochs', '?')}"
        ]
        if info.get("loss") is not None:
            parts.append(f"loss={info['loss']:.3f}")
        if info.get("metric") is not None:
            name, value = info["metric"]
            parts.append(f"{name}={value:.3f}")
        tag = "Poisson" if info.get("algorithm") == ALGORITHM_POISSON else "JSSL"
        return f"[{tag}] " + "  ".join(parts)

    def _finish_training(self, checkpoint, cancelled: bool):
        summary = self._checkpoint_summary()
        if is_poisson_checkpoint(checkpoint):
            cam = checkpoint["camera"]
            self._log(
                f"camera model: offset={cam['offset']:.2f} ADU  "
                f"gain={cam['gain']:.3f} ADU/e-  read noise={cam['read_std']:.3f} e-"
            )
        self.checkpoint_selector.set_session_checkpoint(checkpoint, summary)
        self._on_checkpoint_changed()
        path = self.checkpoint_path_edit.text().strip()
        if path:
            try:
                save_checkpoint(path, checkpoint)
                self._log(f"saved checkpoint to {path}")
            except Exception as exc:
                self._log(f"ERROR saving checkpoint: {exc}")
        status = (
            "Training stopped early -- checkpoint available."
            if cancelled
            else "Training completed."
        )
        self._set_train_status(status, ok=True)

    def _on_train_finished(self, checkpoint):
        self._finish_training(checkpoint, cancelled=False)
        self.train_start_btn.setEnabled(True)
        self.train_cancel_btn.setEnabled(False)
        self._hide_train_force_stop()
        self._set_monitor_running_indicator(False)

    def _on_train_cancelled(self, checkpoint):
        self._finish_training(checkpoint, cancelled=True)
        self.train_start_btn.setEnabled(True)
        self.train_cancel_btn.setEnabled(False)
        self._hide_train_force_stop()
        self._set_monitor_running_indicator(False)

    def _on_train_error(self, message):
        self._log(f"ERROR: {message}")
        short = message.splitlines()[0] if message else ""
        self._set_train_status(
            f"Error: {short[:160]}  (see Monitor tab log)", error=True
        )
        self.train_start_btn.setEnabled(True)
        self.train_cancel_btn.setEnabled(False)
        self._hide_train_force_stop()
        self._set_monitor_running_indicator(False)

    def _on_train_thread_finished(self):
        self._train_worker = None
        self._train_thread = None

    def _cancel_training(self):
        if self._train_worker is not None:
            self._train_worker.cancel()
            self._set_train_status("Cancelling after current step...")
            self.train_cancel_btn.setEnabled(False)
            if not self._train_force_stop_timer.isActive():
                self._train_force_stop_timer.start()

    def _reveal_train_force_stop(self):
        if self._train_worker is not None:
            self._set_train_status(
                "Still cancelling... if this doesn't finish, use Force Stop.",
                error=True,
            )
            self.train_force_stop_btn.setVisible(True)
            self.train_force_stop_btn.setEnabled(True)

    def _hide_train_force_stop(self):
        self._train_force_stop_timer.stop()
        self.train_force_stop_btn.setVisible(False)
        self.train_force_stop_btn.setEnabled(False)

    def _force_stop_training(self):
        if self._train_thread is None:
            self._hide_train_force_stop()
            return
        reply = QMessageBox.warning(
            self,
            "Force Stop",
            "This kills the training thread outright instead of waiting for it to "
            "exit on its own. The in-progress step is abandoned mid-computation and "
            "no checkpoint is recovered from it.\n\nForce stop training?",
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        if reply != QMessageBox.Yes:
            return
        self._train_thread.terminate()
        self._train_thread.wait(3000)
        self._train_worker = None
        self._train_thread = None
        self._hide_train_force_stop()
        self._set_train_status(
            "Force stopped -- no checkpoint recovered.", error=True
        )
        self.train_start_btn.setEnabled(True)
        self.train_cancel_btn.setEnabled(False)
        self._set_monitor_running_indicator(False)

    # ------------------------------------------------------------------
    # Denoise tab
    # ------------------------------------------------------------------

    def _build_denoise_tab(
        self,
        layout,
        SourceSelector,
        RegionSelector,
        ImageOutputSelector,
        BufferProcessingRunner,
        viewer,
    ):
        self.checkpoint_selector = CheckpointSelector(title="Checkpoint")
        self.checkpoint_selector.checkpoint_changed.connect(
            self._on_checkpoint_changed
        )
        self._tta_default_for_poisson = None
        layout.addWidget(self.checkpoint_selector)

        self.source_selector = SourceSelector(title="Source")
        if viewer is not None:
            self.source_selector.set_default_window(viewer)
        self.source_selector.source_changed.connect(self._on_source_changed)
        layout.addWidget(self.source_selector)

        self.region_selector = RegionSelector(title="Region (optional)")
        layout.addWidget(self.region_selector)

        tc_box = QGroupBox("Frame / Plane / Channel")
        tc_form = _tight_form(QFormLayout(tc_box))
        frame_row = QHBoxLayout()
        self.t_start_spin = _ispin(0, 0, 0)
        self.t_end_spin = _ispin(0, 0, 0)
        frame_row.addWidget(QLabel("T:"))
        frame_row.addWidget(self.t_start_spin)
        frame_row.addWidget(QLabel("to"))
        frame_row.addWidget(self.t_end_spin)
        frame_row.addStretch(1)
        tc_form.addRow("Frame range:", frame_row)
        self.z_spin = _ispin(0, 0, 0)
        tc_form.addRow("Z plane:", self.z_spin)
        self.channel_spin = _ispin(0, 0, 0)
        tc_form.addRow("Channel:", self.channel_spin)
        self.z_warning_label = _z_stack_warning_label()
        tc_form.addRow(self.z_warning_label)
        layout.addWidget(tc_box)

        options_box = QGroupBox("Options")
        options_form = _tight_form(QFormLayout(options_box))
        self.tta_cb = QCheckBox(
            "Test-time augmentation (8x dihedral averaging)"
        )
        self.tta_cb.setChecked(True)
        options_form.addRow(self.tta_cb)
        self.device_combo = _device_combo()
        options_form.addRow("Device:", self.device_combo)
        layout.addWidget(options_box)
        layout.addStretch(1)

        self.output_selector = ImageOutputSelector(
            default_title="Denoised", formats=[".tif", ".ims"]
        )
        layout.addWidget(self.output_selector)

        self._runner = BufferProcessingRunner(self.viewer, self.output_selector)

        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        self.status_label = QLabel("Ready")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #888;")
        layout.addWidget(self.status_label)

        buttons = QHBoxLayout()
        buttons.addStretch()
        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self._start)
        buttons.addWidget(self.start_btn)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel)
        buttons.addWidget(self.cancel_btn)
        self.force_stop_btn = QPushButton("Force Stop")
        self.force_stop_btn.setEnabled(False)
        self.force_stop_btn.setVisible(False)
        self.force_stop_btn.clicked.connect(self._force_stop)
        buttons.addWidget(self.force_stop_btn)
        buttons.addStretch()
        layout.addLayout(buttons)

        self._force_stop_timer = QTimer(self)
        self._force_stop_timer.setSingleShot(True)
        self._force_stop_timer.setInterval(5000)
        self._force_stop_timer.timeout.connect(self._reveal_force_stop)

    def _on_checkpoint_changed(self):
        """Reset TTA to the selected algorithm's default only when the
        algorithm changes: 8x is cheap for JSSL (8 passes) but costly for
        the Poisson denoiser (8 x 9 masked passes)."""
        poisson = self.checkpoint_selector.current_is_poisson()
        if poisson != self._tta_default_for_poisson:
            self._tta_default_for_poisson = poisson
            self.tta_cb.setChecked(not poisson)
        self.tta_cb.setToolTip(
            "Poisson checkpoint: 8 x 9 masked network passes per plane."
            if poisson
            else "JSSL checkpoint: 8 network passes per plane."
        )

    def _on_source_changed(self):
        window = self.source_selector.selected_window()
        self.region_selector.set_source(window)
        data, _meta = self.source_selector.get_source()
        if data is not None:
            T, Z, C, _Y, _X = data.shape
            self.t_start_spin.setRange(0, max(0, T - 1))
            self.t_end_spin.setRange(0, max(0, T - 1))
            self.t_end_spin.setValue(max(0, T - 1))
            self.z_spin.setRange(0, max(0, Z - 1))
            self.channel_spin.setRange(0, max(0, C - 1))
            _update_z_stack_warning(self.z_warning_label, Z)
        else:
            self.z_warning_label.setVisible(False)

    def _set_status(self, text, error=False, ok=False):
        self.status_label.setText(text)
        if error:
            self.status_label.setStyleSheet("color: #F44;")
        elif ok:
            self.status_label.setStyleSheet("color: #4A4;")
        else:
            self.status_label.setStyleSheet("color: #888;")

    def _start(self):
        if (
            self._runner.is_running()
            or self._batch_runner.is_running()
            or self._train_thread is not None
        ):
            return

        data, meta = self.source_selector.get_source()
        if data is None:
            self._set_status("No source selected.", error=True)
            return

        T, Z, C, Y, X = data.shape
        bbox = self.region_selector.bbox()
        y0, x0, y1, x1 = bbox if bbox is not None else (0, 0, Y, X)
        t0, t1 = self.t_start_spin.value(), self.t_end_spin.value() + 1
        if t0 >= t1 or t1 > T:
            self._set_status("Invalid T range.", error=True)
            return
        z = self.z_spin.value()
        channels = [self.channel_spin.value()]

        device = _select_device(self.device_combo.currentData())
        denoiser = self.checkpoint_selector.build_denoiser(device)
        if denoiser is None:
            self._set_status("No checkpoint selected.", error=True)
            return
        tta = self.tta_cb.isChecked()

        output_shape = (t1 - t0, 1, len(channels), y1 - y0, x1 - x0)
        base_name = str(meta.get("filename", meta.get("name", "Image")))
        output_meta = dict(meta)
        output_meta["filename"] = f"{base_name}_denoised"
        output_meta["name"] = output_meta["filename"]

        def prepare_for_t(t):
            planes = {
                c: np.asarray(data[t, z, c, y0:y1, x0:x1]) for c in channels
            }
            return {"source_planes": planes}

        def make_worker(frame_data, output_frame_idx):
            worker = DenoiseWorker(
                buffer=self._runner.output_buffer,
                output_t=output_frame_idx,
                source_planes=frame_data["source_planes"],
                denoiser=denoiser,
                tta=tta,
            )
            return worker

        total = (t1 - t0) * max(1, len(channels))
        self.progress_bar.setRange(0, max(1, total))
        self.progress_bar.setValue(0)
        self._set_status("Running...")
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self._hide_force_stop()
        self._set_monitor_running_indicator(True)
        self.log_view.clear()
        self._log(
            f"── denoise {datetime.now():%Y-%m-%d %H:%M:%S} " + "─" * 20 + "\n"
            f"frames={t1 - t0}  z={z}  channels={channels}  tta={tta}  device={device}"
        )

        self._runner.run_frames(
            frame_ts=list(range(t0, t1)),
            output_shape=output_shape,
            output_dtype=np.float32,
            output_meta=output_meta,
            prepare_for_t=prepare_for_t,
            make_worker=make_worker,
            on_progress=self._on_progress,
            on_status=None,
            on_all_finished=self._on_all_finished,
            on_cancelled=self._on_cancelled,
            on_error=self._on_error,
        )

        # run_frames() acquires self.viewer.img_data by default -- swap it
        # for the data the user actually selected via SourceSelector
        # (matches decon_dialog._start's identical swap).
        old_source = self._runner.source_data
        self._runner.source_data = (
            data.acquire() if hasattr(data, "acquire") else data
        )
        if old_source is not None and hasattr(old_source, "release"):
            old_source.release()

    def _cancel(self):
        if self._runner.worker is not None:
            self._runner.cancel()
            self._set_status("Cancelling after current frame...")
            self.cancel_btn.setEnabled(False)
            if not self._force_stop_timer.isActive():
                self._force_stop_timer.start()

    def _reveal_force_stop(self):
        if self._runner.worker is not None:
            self._set_status(
                "Still cancelling... if this doesn't finish, use Force Stop.",
                error=True,
            )
            self.force_stop_btn.setVisible(True)
            self.force_stop_btn.setEnabled(True)

    def _hide_force_stop(self):
        self._force_stop_timer.stop()
        self.force_stop_btn.setVisible(False)
        self.force_stop_btn.setEnabled(False)

    def _force_stop(self):
        if self._runner.worker is None:
            self._hide_force_stop()
            return
        reply = QMessageBox.warning(
            self,
            "Force Stop",
            "This kills the worker thread outright instead of waiting for it to exit "
            "on its own. Any in-progress frame is abandoned mid-computation.\n\n"
            "Force stop the running denoise?",
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        if reply != QMessageBox.Yes:
            return
        thread = self._runner.thread
        if thread is not None:
            thread.terminate()
            thread.wait(3000)
        self._runner.cleanup()
        self._hide_force_stop()
        self._set_status(
            "Force stopped -- state may be inconsistent.", error=True
        )
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self._set_monitor_running_indicator(False)

    def _on_progress(self, done, total):
        self.progress_bar.setValue(done)
        self._set_status(f"Running... {done}/{total}")

    def _on_all_finished(self):
        self._set_status("Completed", ok=True)
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self._hide_force_stop()
        self._set_monitor_running_indicator(False)

    def _on_cancelled(self):
        self._set_status("Cancelled (partial result kept in buffer)")
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self._hide_force_stop()
        self._set_monitor_running_indicator(False)

    def _on_error(self, message):
        self._log(f"ERROR: {message}")
        short = message.splitlines()[0] if message else ""
        self._set_status(
            f"Error: {short[:160]}  (see Monitor tab log)", error=True
        )
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self._hide_force_stop()
        self._set_monitor_running_indicator(False)

    # ------------------------------------------------------------------
    # Batch tab -- reuses this dialog's Denoise-tab settings (checkpoint,
    # source, region, T/Z/channel, TTA, device) across every checked file,
    # exactly as decon_dialog's batch tab reuses its Input/Solver settings.
    # ------------------------------------------------------------------

    def _build_batch_tab(self, FlaggableFileListWidget):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        intro = QLabel(
            "Applies the Denoise tab's settings above (checkpoint, Region/Z-plane/"
            "T-range/channel, TTA, device) unchanged to every checked file below. "
            "Files that can't fit those settings (too few channels/timepoints/planes) "
            "are auto-flagged and unchecked, but can be re-checked manually."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(intro)

        folder_row = QHBoxLayout()
        load_folder_btn = QPushButton("Load Folder...")
        load_folder_btn.clicked.connect(self._browse_batch_folder)
        folder_row.addWidget(load_folder_btn)
        revalidate_btn = QPushButton("Re-validate")
        revalidate_btn.clicked.connect(self._validate_batch_files)
        folder_row.addWidget(revalidate_btn)
        folder_row.addStretch(1)
        layout.addLayout(folder_row)

        self.batch_file_list = FlaggableFileListWidget()
        self.batch_file_list.filesChanged.connect(self._on_batch_files_changed)
        self.batch_file_list.checkStateChanged.connect(
            self._update_batch_status_label
        )
        layout.addWidget(self.batch_file_list, 1)

        self.batch_status_label = QLabel("No files loaded.")
        self.batch_status_label.setWordWrap(True)
        self.batch_status_label.setStyleSheet("color: #888;")
        layout.addWidget(self.batch_status_label)

        output_row = QHBoxLayout()
        output_row.addWidget(QLabel("Output format:"))
        self.batch_output_format_combo = QComboBox()
        from pyvistra.io import get_output_format

        for ext in (".tif", ".ims"):
            fmt = get_output_format(ext)
            label = fmt[0] if fmt is not None else ext
            self.batch_output_format_combo.addItem(label, ext)
        output_row.addWidget(self.batch_output_format_combo)
        output_row.addStretch(1)
        layout.addLayout(output_row)

        self.batch_progress_bar = QProgressBar()
        layout.addWidget(self.batch_progress_bar)

        buttons = QHBoxLayout()
        buttons.addStretch()
        self.batch_start_btn = QPushButton("Start Batch")
        self.batch_start_btn.clicked.connect(self._start_batch)
        buttons.addWidget(self.batch_start_btn)
        self.batch_cancel_btn = QPushButton("Cancel Batch")
        self.batch_cancel_btn.setEnabled(False)
        self.batch_cancel_btn.clicked.connect(self._cancel_batch)
        buttons.addWidget(self.batch_cancel_btn)
        buttons.addStretch()
        layout.addLayout(buttons)

        self.tabs.addTab(page, "Batch")

        self._batch_runner = BatchDenoiseRunner()
        self._batch_runner.file_progress.connect(self._on_batch_file_progress)
        self._batch_runner.progress.connect(self._on_progress)
        self._batch_runner.log_line.connect(self._on_log_line)
        self._batch_runner.file_error.connect(self._on_batch_file_error)
        self._batch_runner.all_finished.connect(self._on_batch_all_finished)
        self._batch_runner.cancelled.connect(self._on_batch_cancelled)

    def _set_batch_status(self, text, error=False, ok=False):
        self.batch_status_label.setText(text)
        if error:
            self.batch_status_label.setStyleSheet("color: #F44;")
        elif ok:
            self.batch_status_label.setStyleSheet("color: #4A4;")
        else:
            self.batch_status_label.setStyleSheet("color: #888;")

    def _browse_batch_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Load Folder")
        if not folder:
            return
        self._batch_folder = folder
        self.batch_file_list.set_folder(folder)

    def _on_batch_files_changed(self):
        paths = self.batch_file_list.paths()
        if paths:
            try:
                self._batch_folder = (
                    os.path.commonpath(paths)
                    if len(paths) > 1
                    else os.path.dirname(paths[0])
                )
            except ValueError:
                self._batch_folder = os.path.dirname(paths[0])
        self._validate_batch_files()

    def _update_batch_status_label(self):
        total = len(self.batch_file_list.paths())
        checked = len(self.batch_file_list.checked_paths())
        self._set_batch_status(
            f"{checked} of {total} file(s) will be processed."
        )

    def _reference_requirements(self):
        data, meta = self.source_selector.get_source()
        if data is None:
            return None
        T, Z, C, Y, X = data.shape
        bbox = self.region_selector.bbox()
        y0, x0, y1, x1 = bbox if bbox is not None else (0, 0, Y, X)
        t0, t1 = self.t_start_spin.value(), self.t_end_spin.value() + 1
        z = self.z_spin.value()
        c = self.channel_spin.value()
        return dict(
            t0=t0,
            t1=t1,
            z=z,
            y0=y0,
            x0=x0,
            y1=y1,
            x1=x1,
            channel=c,
            min_t=t1,
            min_z=z + 1,
            min_c=c + 1,
        )

    def _validate_batch_files(self):
        ref = self._reference_requirements()
        if ref is None:
            self._set_batch_status(
                "No source selected in Denoise tab -- can't validate.",
                error=True,
            )
            return

        from pyvistra.io import load_image

        for path in self.batch_file_list.paths():
            try:
                data, meta = load_image(path)
            except Exception as exc:
                self.batch_file_list.set_flag(
                    path, "error", f"Failed to load: {exc}"
                )
                continue
            try:
                T, Z, C, Y, X = data.shape
                reasons = []
                if T < ref["min_t"]:
                    reasons.append(f"T={T} < required {ref['min_t']}")
                if Z < ref["min_z"]:
                    reasons.append(f"Z={Z} < required {ref['min_z']}")
                if C < ref["min_c"]:
                    reasons.append(
                        f"only {C} channel(s), need >= {ref['min_c']}"
                    )
                if Y < ref["y1"]:
                    reasons.append(f"Y={Y} < required {ref['y1']}")
                if X < ref["x1"]:
                    reasons.append(f"X={X} < required {ref['x1']}")
                if reasons:
                    self.batch_file_list.set_flag(
                        path, "error", "; ".join(reasons)
                    )
                elif Z > ref["min_z"]:
                    self.batch_file_list.set_flag(
                        path,
                        "warn",
                        f"Z-stack ({Z} planes) -- only z={ref['z']} is denoised; "
                        "both denoisers are 2D/2D+time only.",
                    )
                else:
                    self.batch_file_list.set_flag(path, "ok")
            finally:
                if hasattr(data, "release"):
                    try:
                        data.release()
                    except Exception:
                        pass

        self._update_batch_status_label()

    def _start_batch(self):
        if (
            self._runner.is_running()
            or self._batch_runner.is_running()
            or self._train_thread is not None
        ):
            return

        checked = self.batch_file_list.checked_paths()
        if not checked:
            self._set_batch_status("No files checked.", error=True)
            return

        ref = self._reference_requirements()
        if ref is None:
            self._set_batch_status(
                "No source selected in Denoise tab.", error=True
            )
            return

        device = _select_device(self.device_combo.currentData())
        denoiser = self.checkpoint_selector.build_denoiser(device)
        if denoiser is None:
            self._set_batch_status("No checkpoint selected.", error=True)
            return

        region_params = dict(
            t0=ref["t0"],
            t1=ref["t1"],
            z0=ref["z"],
            y0=ref["y0"],
            x0=ref["x0"],
            y1=ref["y1"],
            x1=ref["x1"],
            channels=[ref["channel"]],
        )
        out_Y, out_X = ref["y1"] - ref["y0"], ref["x1"] - ref["x0"]
        output_shape = (ref["t1"] - ref["t0"], 1, 1, out_Y, out_X)

        _data, meta = self.source_selector.get_source()
        output_meta_template = dict(meta) if meta else {}

        output_ext = self.batch_output_format_combo.currentData()
        folder = self._batch_folder or os.path.dirname(checked[0])
        log_path = os.path.join(
            folder, f"batch_denoise_{datetime.now():%Y%m%d_%H%M%S}.log"
        )

        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.batch_progress_bar.setRange(0, max(1, len(checked)))
        self.batch_progress_bar.setValue(0)
        self._set_batch_status("Running batch...")
        self.batch_start_btn.setEnabled(False)
        self.batch_cancel_btn.setEnabled(True)
        self._set_monitor_running_indicator(True)
        self.log_view.clear()

        self._batch_runner.start(
            file_paths=checked,
            denoiser=denoiser,
            tta=self.tta_cb.isChecked(),
            region_params=region_params,
            output_shape=output_shape,
            output_meta_template=output_meta_template,
            output_ext=output_ext,
            log_path=log_path,
        )

    def _cancel_batch(self):
        if self._batch_runner.is_running():
            self._batch_runner.cancel()
            self._set_batch_status("Cancelling after current file...")
            self.batch_cancel_btn.setEnabled(False)

    def _on_batch_file_progress(self, done, total):
        self.batch_progress_bar.setRange(0, max(1, total))
        self.batch_progress_bar.setValue(done)
        if done < total:
            self._set_batch_status(f"File {done + 1}/{total}...")

    def _on_batch_file_error(self, path, message):
        short = message.splitlines()[0] if message else ""
        self._set_batch_status(
            f"Error on {os.path.basename(path)}: {short[:160]}", error=True
        )

    def _on_batch_all_finished(self):
        self._set_batch_status("Batch completed.", ok=True)
        self.batch_progress_bar.setValue(self.batch_progress_bar.maximum())
        self.batch_start_btn.setEnabled(True)
        self.batch_cancel_btn.setEnabled(False)
        self._set_monitor_running_indicator(False)

    def _on_batch_cancelled(self):
        self._set_batch_status("Batch cancelled (already-saved files kept).")
        self.batch_start_btn.setEnabled(True)
        self.batch_cancel_btn.setEnabled(False)
        self._set_monitor_running_indicator(False)

    # ------------------------------------------------------------------
    # Monitor tab -- shared loss/mu_mse plot + run log, reused across
    # Train/Denoise/Batch (see decon_dialog's identical reuse).
    # ------------------------------------------------------------------

    def _build_monitor_tab(self, ConvergencePlotWidget):
        self.convergence_plot = ConvergencePlotWidget()
        self.convergence_plot.set_labels(
            x_label="Epoch", y_label="loss", y_label_right="mu_mse"
        )

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(2000)
        self.log_view.setMinimumHeight(80)

        monitor_tab = QWidget()
        monitor_layout = QVBoxLayout(monitor_tab)
        monitor_layout.setContentsMargins(6, 6, 6, 6)
        monitor_splitter = QSplitter(Qt.Vertical)
        monitor_splitter.addWidget(self.convergence_plot)
        monitor_splitter.addWidget(self.log_view)
        monitor_splitter.setStretchFactor(0, 2)
        monitor_splitter.setStretchFactor(1, 1)
        monitor_layout.addWidget(monitor_splitter)
        self._monitor_tab_index = self.tabs.addTab(monitor_tab, "Monitor")

    def _set_monitor_running_indicator(self, running):
        self.tabs.setTabText(
            self._monitor_tab_index, "Monitor ●" if running else "Monitor"
        )

    def _log(self, text):
        self.log_view.appendPlainText(text)

    def _on_log_line(self, text):
        self.log_view.appendPlainText(text)

    # ------------------------------------------------------------------
    # Close
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        if self._train_thread is not None:
            self._cancel_training()
            event.ignore()
            return
        if self._runner.worker is not None:
            self._cancel()
            event.ignore()
            return
        if self._batch_runner.is_running():
            self._cancel_batch()
            event.ignore()
            return
        super().closeEvent(event)
