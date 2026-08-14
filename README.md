# jssl-denoise

Joint self-supervised blind denoising and noise estimation for microscopy
timelapses, implemented in PyTorch after
[Ollion et al., 2021](https://arxiv.org/abs/2102.08023).

Given a single noisy `(T, H, W)` timelapse stack (no clean reference needed),
the method trains a paired **D-Net** (denoiser) and **N-Net** (per-pixel
noise-std estimator) using masked self-supervision across frames, then
denoises full-size images with optional 8-way dihedral test-time
augmentation.

## Installation

```bash
pip install -e .            # core (torch, numpy)
pip install -e ".[examples]"  # + tifffile, pillow, for the example scripts
pip install -e ".[dev]"       # + pytest
```

Requires Python >= 3.10.

## Usage

### Train on a timelapse stack

```bash
python examples/train_example.py example/de_gems.tif checkpoints/de_gems.pt \
    --epochs 60 --steps-per-epoch 100
```

```python
from jssl_denoise import ConsoleCallback, TrainingConfig, save_checkpoint
from jssl_denoise.training import Trainer

config = TrainingConfig(epochs=60, steps_per_epoch=100, device="mps")
trainer = Trainer(config)
checkpoint = trainer.fit(stack, callback=ConsoleCallback())  # stack: (T, H, W) ndarray
save_checkpoint("checkpoints/de_gems.pt", checkpoint)
```

### Denoise with a trained checkpoint

```bash
python examples/denoise_example.py checkpoints/de_gems.pt example/de_gems.tif --output-dir out/
```

```python
from jssl_denoise import Denoiser

denoiser = Denoiser.load("checkpoints/de_gems.pt")
denoised, noise_std_map = denoiser.denoise(image, tta=True)  # image: 2D ndarray
```

## Package layout

| Module | Responsibility |
|---|---|
| `networks.py` | D-Net (U-Net denoiser) and N-Net (noise-std head) architectures |
| `masking.py` | Random pixel masking used for self-supervised training |
| `losses.py` | Masked beta-NLL Gaussian loss (Seitzer et al., 2022) |
| `patches.py` | Tiling/patch sampling for training |
| `normalization.py` | `RobustNormalizer` — percentile-based intensity normalization |
| `training.py` | `Trainer` — training loop, device selection |
| `inference.py` | `Denoiser` — full-size inference with dihedral TTA |
| `checkpoint.py` | Save/load trained D-Net/N-Net pairs |
| `callbacks.py` | Training progress callbacks (`ConsoleCallback`) |
| `config.py` | `TrainingConfig` hyperparameters |

## Testing

```bash
pytest
```
