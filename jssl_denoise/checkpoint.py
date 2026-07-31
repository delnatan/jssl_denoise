from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: str | Path, checkpoint: dict[str, Any]) -> None:
    """Saves a checkpoint dict produced by `Trainer.fit` (state dicts,
    normalizer stats, and training config) to a single file.
    """
    torch.save(checkpoint, path)


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)
