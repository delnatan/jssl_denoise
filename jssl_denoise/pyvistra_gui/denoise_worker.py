"""Denoising worker -- pyvistra plugin for jssl_denoise.

One :class:`DenoiseWorker` instance handles one T-frame (all selected
channels), constructed by pyvistra's ``BufferProcessingRunner.run_frames``
via a ``make_worker(frame_data, output_frame_idx)`` closure -- the exact
convention resolvde's ``DeconvolutionWorker`` uses (see
``denoise_dialog.py``).
"""

from __future__ import annotations

import numpy as np
from qtpy.QtCore import QObject, Signal

from ._device_cleanup import release_device_cache


class DenoiseWorker(QObject):
    """Denoise one T-frame (all selected channels) into an output buffer.

    ``buffer`` is a Writable5D sized for the (possibly ROI-cropped) source
    shape -- see ``DenoiseDialog._start``. Unlike resolvde's deconvolution
    worker, there's no Z dimension here: `jssl_denoise`'s D-Net operates on
    single 2D planes, so every write goes through the 2D branch of
    ``_write_plane``.
    """

    progress = Signal(int, int)  # channels done, total channels
    finished = Signal()
    cancelled = Signal()
    error = Signal(str)

    def __init__(self, buffer, output_t, source_planes, denoiser, tta):
        """
        Args:
            buffer: Writable5D output buffer.
            output_t: T index in ``buffer`` this worker writes into.
            source_planes: dict[channel_idx -> np.ndarray (Y, X)], observed
                data, already ROI/T-cropped, at a single Z plane.
            denoiser: a ready `jssl_denoise.inference.Denoiser` (already
                `.eval()`'d and on its target device).
            tta: bool, forwarded to `Denoiser.denoise`.
        """
        super().__init__()
        self._buffer = buffer
        self._output_t = output_t
        self._source_planes = source_planes
        self._denoiser = denoiser
        self._tta = tta
        self._cancel_requested = False
        # Mirrors DeconvolutionWorker: `buffer`'s C axis is sized by the
        # *count* of selected channels, not the source's real channel index.
        self._channel_positions = {
            c: i for i, c in enumerate(sorted(source_planes))
        }

    def cancel(self):
        self._cancel_requested = True

    def run(self):
        try:
            channels = sorted(self._source_planes)
            total = max(1, len(channels))
            for i, c in enumerate(channels):
                if self._cancel_requested:
                    self.cancelled.emit()
                    return
                denoised, _noise_std = self._denoiser.denoise(
                    self._source_planes[c], tta=self._tta
                )
                self._write_plane(c, denoised)
                self.progress.emit(i + 1, total)

            self.finished.emit()

        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            # TTA runs 8 forward passes per channel -- release MPS/CUDA's
            # cache so repeated Denoise/Batch runs in a long-lived GUI
            # session don't accumulate pinned device memory (see
            # _device_cleanup.py; same pattern as resolvde's
            # DeconvolutionWorker._release_device_cache).
            release_device_cache(self._denoiser.device)

    def _write_plane(self, channel_idx, plane):
        plane = np.asarray(plane, dtype=self._buffer.dtype)
        c = self._channel_positions[channel_idx]
        self._buffer[self._output_t, 0, c, :, :] = plane
