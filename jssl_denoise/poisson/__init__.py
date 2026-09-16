"""Bayesian blind-spot denoising under a Poisson-Gaussian camera model,
after de Wolf, Nonnekens & Smal (ISBI 2026), extended with camera gain,
offset and read noise. Independent of the Ollion D-Net/N-Net method in the
parent package; shares its patch sampling, masking, normalization and U-Net.
"""

from .camera import CameraModel, estimate_camera_model
from .config import PoissonTrainingConfig
from .inference import PoissonDenoiser
from .network import GammaPriorNet
from .training import PoissonTrainer

__all__ = [
    "CameraModel",
    "GammaPriorNet",
    "PoissonDenoiser",
    "PoissonTrainer",
    "PoissonTrainingConfig",
    "estimate_camera_model",
]
