"""Denoises every frame of a raw-ADU TIFF timelapse with a trained Poisson
denoiser checkpoint, saving the posterior mean, its posterior standard
deviation, and frame-0 preview PNGs.

Usage:
    python examples/denoise_poisson_example.py checkpoints/de_gems_poisson.pt example/de_gems.tif --output-dir out/poisson
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

from jssl_denoise.poisson import PoissonDenoiser


def _to_uint8(image: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(image, [0.5, 99.5])
    scaled = np.clip((image - lo) / max(hi - lo, 1e-6), 0, 1)
    return (scaled * 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("input", type=Path, help="path to a raw-ADU (T,H,W) TIFF stack")
    parser.add_argument("--output-dir", type=Path, default=Path("out/poisson"))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--tta", action="store_true", help="average over 8 dihedral transforms (8x slower)")
    args = parser.parse_args()

    stack = tifffile.imread(args.input)
    denoiser = PoissonDenoiser.load(args.checkpoint, device=args.device)

    n_frames = stack.shape[0]
    denoised_stack = np.empty(stack.shape, dtype=np.float32)
    std_stack = np.empty(stack.shape, dtype=np.float32)
    for t in range(n_frames):
        denoised_stack[t], std_stack[t] = denoiser.denoise(stack[t], tta=args.tta)
        print(f"denoised frame {t + 1}/{n_frames}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_to_uint8(stack[0])).save(args.output_dir / "noisy_frame0.png")
    Image.fromarray(_to_uint8(denoised_stack[0])).save(args.output_dir / "denoised_frame0.png")
    tifffile.imwrite(args.output_dir / "denoised.tif", denoised_stack)
    tifffile.imwrite(args.output_dir / "posterior_std.tif", std_stack)
    print(f"saved denoised.tif, posterior_std.tif and frame-0 previews to {args.output_dir}")


if __name__ == "__main__":
    main()
