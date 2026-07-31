"""Shared MPS/CUDA cache-release helper for the pyvistra plugin workers.

Neither MPS's nor CUDA's caching allocator returns freed-but-unused blocks
to the driver on its own -- they stay reserved for reuse by that process.
On Apple Silicon's unified memory that reservation competes directly with
system RAM, so a long training run or a run of many TTA-augmented denoise
calls can otherwise leave GPU memory pinned for the rest of a long-lived
GUI session, well past the point the tensors that used it are gone.
Cheap and safe to call often. Same pattern, same justification, as
resolvde's ``DeconvolutionWorker._release_device_cache`` in the sibling
pyvistra plugin -- factored out here since both `TrainingWorker` and
`DenoiseWorker` need it.
"""

from __future__ import annotations


def release_device_cache(device) -> None:
    if device is None:
        return
    try:
        import torch

        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()
    except Exception:
        pass
