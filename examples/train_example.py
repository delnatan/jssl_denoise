"""Trains a D-Net/N-Net pair on one of the example timelapse TIFFs.

Usage:
    python examples/train_example.py example/de_gems.tif checkpoints/de_gems.pt --epochs 60 --steps-per-epoch 100
"""

from __future__ import annotations

import argparse
from pathlib import Path

import tifffile

from jssl_denoise import ConsoleCallback, TrainingConfig, save_checkpoint
from jssl_denoise.training import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="path to a (T,H,W) TIFF stack")
    parser.add_argument("output", type=Path, help="path to save the trained checkpoint (.pt)")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--tile-size", type=int, default=64)
    parser.add_argument("--tiles-per-batch", type=int, default=32)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    stack = tifffile.imread(args.input)
    print(f"loaded {args.input}: shape={stack.shape} dtype={stack.dtype}")

    config = TrainingConfig(
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        tile_size=args.tile_size,
        tiles_per_batch=args.tiles_per_batch,
        device=args.device,
        seed=args.seed,
    )
    trainer = Trainer(config)
    checkpoint = trainer.fit(stack, callback=ConsoleCallback())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(args.output, checkpoint)
    print(f"saved checkpoint to {args.output}")


if __name__ == "__main__":
    main()
