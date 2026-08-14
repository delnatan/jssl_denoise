"""Denoises every frame of an example TIFF timelapse with a trained
checkpoint and reports before/after quality proxies (no ground truth is
available, so the temporal mean across the stack is used as an
evaluation-only pseudo-clean reference -- never used for training).

Usage:
    python examples/denoise_example.py checkpoints/de_gems.pt example/de_gems.tif --output-dir out/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

from jssl_denoise import Denoiser


def _to_uint8(image: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(image, [0.5, 99.5])
    scaled = np.clip((image - lo) / max(hi - lo, 1e-6), 0, 1)
    return (scaled * 255).astype(np.uint8)


def _psnr(prediction: np.ndarray, reference: np.ndarray) -> float:
    mse = float(np.mean((prediction.astype(np.float64) - reference.astype(np.float64)) ** 2))
    if mse == 0:
        return float("inf")
    data_range = float(reference.max() - reference.min())
    return 10.0 * np.log10((data_range**2) / mse)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("input", type=Path, help="path to the (T,H,W) TIFF stack used for training")
    parser.add_argument("--output-dir", type=Path, default=Path("out"))
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    stack = tifffile.imread(args.input)
    pseudo_clean = stack.mean(axis=0)

    denoiser = Denoiser.load(args.checkpoint, device=args.device)

    n_frames = stack.shape[0]
    denoised_stack = np.empty_like(stack, dtype=np.float32)
    noise_std_stack = np.empty_like(stack, dtype=np.float32)
    for t in range(n_frames):
        denoised, noise_std_map = denoiser.denoise(stack[t], tta=True)
        denoised_stack[t] = denoised
        noise_std_stack[t] = noise_std_map
        print(f"denoised frame {t + 1}/{n_frames}")

    psnr_noisy = np.mean([_psnr(stack[t], pseudo_clean) for t in range(n_frames)])
    psnr_denoised = np.mean([_psnr(denoised_stack[t], pseudo_clean) for t in range(n_frames)])
    print(f"mean PSNR vs. temporal-mean proxy: noisy={psnr_noisy:.2f} dB  denoised={psnr_denoised:.2f} dB")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_to_uint8(stack[0])).save(args.output_dir / "noisy_frame0.png")
    Image.fromarray(_to_uint8(denoised_stack[0])).save(args.output_dir / "denoised_frame0.png")
    print(f"saved noisy/denoised frame-0 preview PNGs to {args.output_dir}")

    tifffile.imwrite(args.output_dir / "denoised.tif", denoised_stack)
    tifffile.imwrite(args.output_dir / "noise_std_map.tif", noise_std_stack)
    print(f"saved denoised.tif and noise_std_map.tif (float32, {n_frames} frames) to {args.output_dir}")


if __name__ == "__main__":
    main()
