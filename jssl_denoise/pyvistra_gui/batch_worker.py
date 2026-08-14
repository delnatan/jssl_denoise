"""Batch denoising orchestrator -- pyvistra plugin for jssl_denoise.

Structural copy of resolvde's ``BatchDeconvolveRunner``: drives the same
:class:`~.denoise_worker.DenoiseWorker` the single-image path uses, one
T-frame at a time, with an outer loop over a list of *files* -- each gets
its own freshly loaded source and its own output buffer, saved to disk and
released before the next file starts. Not built on pyvistra's
``BufferProcessingRunner`` for the same reason resolvde's isn't: that class
hardcodes its source as ``self.viewer.img_data`` (one already-open window),
which doesn't fit "load a fresh file per batch item".

Unlike deconvolution's batch runner, there's no PSF to load per run and no
per-file wavelength-homogeneity check -- a trained checkpoint has no PSF
and no emission-wavelength metadata to compare against, only the source's
shape (checked by the dialog's own validation, mirroring
``decon_dialog._validate_batch_files``).
"""

from __future__ import annotations

import os
import time

import numpy as np
from qtpy.QtCore import QObject, QThread, Signal

from .denoise_worker import DenoiseWorker


class BatchDenoiseRunner(QObject):
    """Sequentially denoise every file in ``file_paths`` with one
    `Denoiser` and one resolved settings dict, saving each result next to
    its source.

    ``region_params`` (identical crop applied to every file -- homogeneity
    is checked by the caller before ``start()``):
        ``t0, t1, z0, y0, x0, y1, x1, channels`` (list[int]).

    ``output_shape``/``output_meta_template`` are precomputed once by the
    caller (same for every file, since the crop and channel set are fixed)
    -- ``output_meta_template`` gets its ``filename``/``name`` overwritten
    per file.
    """

    file_progress = Signal(int, int)  # files done, total
    progress = Signal(int, int)  # channel progress within current frame
    log_line = Signal(str)
    file_error = Signal(str, str)  # path, message
    all_finished = Signal()
    cancelled = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.thread = None
        self.worker = None
        self._batch = None
        self._log_fh = None

    def is_running(self):
        return self._batch is not None

    def start(
        self,
        *,
        file_paths,
        denoiser,
        tta,
        region_params,
        output_shape,
        output_meta_template,
        output_ext,
        log_path,
    ):
        self._log_fh = open(log_path, "a", encoding="utf-8")
        self._write_run_header(file_paths, region_params, output_ext)

        self._batch = dict(
            file_paths=list(file_paths),
            file_idx=0,
            denoiser=denoiser,
            tta=tta,
            region_params=region_params,
            output_shape=output_shape,
            output_meta_template=output_meta_template,
            output_ext=output_ext,
            cancel_requested=False,
        )
        self.file_progress.emit(0, len(file_paths))
        self._start_next_file()

    def cancel(self):
        if self._batch is not None:
            self._batch["cancel_requested"] = True
        if self.worker is not None:
            self.worker.cancel()

    # ------------------------------------------------------------------
    # File loop
    # ------------------------------------------------------------------

    def _start_next_file(self):
        b = self._batch
        if b["cancel_requested"]:
            self._finish(cancelled=True)
            return
        if b["file_idx"] >= len(b["file_paths"]):
            self._finish(all_done=True)
            return

        from pyvistra.io import load_image

        path = b["file_paths"][b["file_idx"]]
        b["current_path"] = path
        b["file_start_time"] = time.monotonic()

        try:
            data, meta = load_image(path)
        except Exception as exc:
            self._log(f"[{os.path.basename(path)}] ERROR loading: {exc}")
            self.file_error.emit(path, str(exc))
            self._advance_file()
            return

        from pyvistra.io import ImageBuffer

        b["current_source"] = data
        b["current_meta"] = meta
        output_meta = dict(b["output_meta_template"])
        base_name = os.path.splitext(os.path.basename(path))[0]
        output_meta["filename"] = f"{base_name}_denoised"
        output_meta["name"] = output_meta["filename"]
        b["current_output_meta"] = output_meta
        b["current_buffer"] = ImageBuffer(
            shape=b["output_shape"], dtype=np.float32, metadata=output_meta
        )

        region = b["region_params"]
        b["t_list"] = list(range(region["t0"], region["t1"]))
        b["t_idx"] = 0
        b["frame_error"] = None

        self._log(f"── {os.path.basename(path)} " + "─" * 20)
        self._run_next_frame()

    def _advance_file(self):
        b = self._batch
        b["file_idx"] += 1
        self.file_progress.emit(b["file_idx"], len(b["file_paths"]))
        self._start_next_file()

    # ------------------------------------------------------------------
    # Per-T-frame worker chaining (mirrors BufferProcessingRunner._run_next_frame)
    # ------------------------------------------------------------------

    def _run_next_frame(self):
        b = self._batch
        if b["cancel_requested"]:
            self._finish_current_file(cancelled=True)
            return

        region = b["region_params"]
        data = b["current_source"]
        t = b["t_list"][b["t_idx"]]

        planes = {}
        for c in region["channels"]:
            planes[c] = np.asarray(
                data[
                    t,
                    region["z0"],
                    c,
                    region["y0"] : region["y1"],
                    region["x0"] : region["x1"],
                ]
            )

        worker = DenoiseWorker(
            buffer=b["current_buffer"],
            output_t=b["t_idx"],
            source_planes=planes,
            denoiser=b["denoiser"],
            tta=b["tta"],
        )
        self.worker = worker
        self.thread = QThread()
        worker.moveToThread(self.thread)

        self.thread.started.connect(worker.run)
        worker.progress.connect(self.progress)
        worker.finished.connect(self._on_frame_finished)
        worker.cancelled.connect(self._on_frame_cancelled)
        worker.error.connect(self._on_frame_error)

        worker.finished.connect(self.thread.quit)
        worker.cancelled.connect(self.thread.quit)
        worker.error.connect(self.thread.quit)

        worker.finished.connect(worker.deleteLater)
        worker.cancelled.connect(worker.deleteLater)
        worker.error.connect(worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(self._on_frame_thread_finished)

        self.thread.start()

    def _on_frame_finished(self):
        self._batch["t_idx"] += 1

    def _on_frame_cancelled(self):
        self._batch["cancel_requested"] = True

    def _on_frame_error(self, message):
        self._batch["frame_error"] = message

    def _on_frame_thread_finished(self):
        self.worker = None
        self.thread = None
        b = self._batch

        if b["frame_error"]:
            message = b["frame_error"]
            path = b["current_path"]
            self._log(f"[{os.path.basename(path)}] ERROR: {message}")
            self.file_error.emit(path, message)
            self._finish_current_file(errored=True)
            return
        if b["cancel_requested"]:
            self._finish_current_file(cancelled=True)
            return
        if b["t_idx"] >= len(b["t_list"]):
            self._finish_current_file(completed=True)
            return
        self._run_next_frame()

    def _finish_current_file(
        self, completed=False, cancelled=False, errored=False
    ):
        from pyvistra.io import get_output_format

        b = self._batch
        path = b["current_path"]
        buffer = b.pop("current_buffer", None)
        source = b.pop("current_source", None)

        if completed and buffer is not None:
            out_path = self._output_path(path, b["output_ext"])
            try:
                _label, saver = get_output_format(b["output_ext"])
                saver(out_path, buffer, b["current_output_meta"])
                elapsed = time.monotonic() - b["file_start_time"]
                self._log(
                    f"[{os.path.basename(path)}] done in {elapsed:.1f}s -> {out_path}"
                )
            except Exception as exc:
                self._log(f"[{os.path.basename(path)}] ERROR saving: {exc}")
                self.file_error.emit(path, str(exc))

        if buffer is not None:
            buffer.close()
        if source is not None and hasattr(source, "release"):
            try:
                source.release()
            except Exception:
                pass

        if cancelled:
            self._finish(cancelled=True)
            return
        self._advance_file()

    def _finish(self, all_done=False, cancelled=False):
        if self._log_fh is not None:
            self._log_fh.write(
                f"── batch {'cancelled' if cancelled else 'finished'} ──\n"
            )
            self._log_fh.close()
            self._log_fh = None
        self._batch = None
        if cancelled:
            self.cancelled.emit()
        else:
            self.all_finished.emit()

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log(self, text):
        self.log_line.emit(text)
        if self._log_fh is not None:
            self._log_fh.write(text + "\n")
            self._log_fh.flush()

    def _write_run_header(self, file_paths, region_params, output_ext):
        from datetime import datetime

        lines = [
            f"══ batch run {datetime.now():%Y-%m-%d %H:%M:%S} " + "═" * 20,
            f"output_ext={output_ext}",
            f"region: t={region_params['t0']}:{region_params['t1']} "
            f"z={region_params['z0']} "
            f"y={region_params['y0']}:{region_params['y1']} "
            f"x={region_params['x0']}:{region_params['x1']} "
            f"channels={region_params['channels']}",
            f"{len(file_paths)} file(s):",
        ]
        lines.extend(f"  {p}" for p in file_paths)
        lines.append("═" * 45)
        for line in lines:
            self._log(line)

    def _output_path(self, source_path, ext):
        stem, _ = os.path.splitext(source_path)
        return f"{stem}_denoised{ext}"
