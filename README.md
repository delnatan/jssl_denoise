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
pip install -e ".[gui]"       # + pyvistra plugin (Qt-based viewer integration)
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

### pyvistra viewer plugin

Installing the `gui` extra registers an "Image > Denoising > Denoise..."
menu item in pyvistra that wraps training and inference in a dialog. The
plugin is discovered lazily via the `pyvistra.plugins` entry point and only
imports Qt/torch when the menu is built, not at `import jssl_denoise` time.

### Bayesian Poisson–Gaussian denoiser (`jssl_denoise.poisson`)

A separate method after
[de Wolf, Nonnekens & Smal, ISBI 2026](https://doi.org/10.1109/ISBI61048.2026.11516009),
extended with camera read noise. A blind-spot U-Net predicts a per-pixel
Gamma prior over the photon rate; training maximizes the exact
Poisson–Gaussian marginal likelihood, and denoising returns the posterior
mean, which folds each pixel's own observed value back in.

The camera model (offset, gain in ADU/e⁻, read noise in e⁻) is fitted from the
raw images before training and then frozen. Gain and read noise cannot be
learned jointly with the prior, since a free prior variance absorbs them. Pass
the camera's black level as `offset` for a physically meaningful read-noise
estimate. Input must be raw ADU (not background-subtracted or rescaled).

```bash
python examples/train_poisson_example.py example/de_gems.tif checkpoints/de_gems_poisson.pt --offset 100
python examples/denoise_poisson_example.py checkpoints/de_gems_poisson.pt example/de_gems.tif --output-dir out/poisson
```

```python
from jssl_denoise.poisson import PoissonDenoiser, PoissonTrainer, PoissonTrainingConfig

checkpoint = PoissonTrainer(PoissonTrainingConfig(offset=100.0)).fit(stack)
print(checkpoint["camera"])  # {'offset': ..., 'gain': ..., 'read_std': ...}

denoiser = PoissonDenoiser.from_checkpoint(checkpoint, device)
denoised, posterior_std = denoiser.denoise(image)  # both in ADU
```

Inference runs one masked pass per grid phase (`mask_spacing**2`, default 9)
so the prior never sees the pixel it is combined with; `tta=True` multiplies
that by 8.

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
| `poisson/camera.py` | `CameraModel` and `estimate_camera_model` — gain/offset/read noise from raw images |
| `poisson/likelihood.py` | Exact Poisson–Gaussian marginal likelihood and posterior moments under a Gamma prior |
| `poisson/network.py` | `GammaPriorNet` — U-Net predicting the per-pixel Gamma prior |
| `poisson/training.py` | `PoissonTrainer` |
| `poisson/inference.py` | `PoissonDenoiser` — blind-spot prior + Bayesian posterior mean |
| `pyvistra_gui/` | Qt dialog and workers for the pyvistra plugin |

## Testing

```bash
pytest
```
