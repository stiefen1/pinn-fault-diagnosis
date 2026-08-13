"""Factories for model-training objects built from config."""

from typing import Any

import torch
from torch import nn
from torch.optim import Adam, AdamW, SGD, Optimizer, lr_scheduler

ACTIVATION_CLASSES: dict[str, type[nn.Module] | None] = {
	"relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
	"tanh": nn.Tanh,
	"gelu": nn.GELU,
	"silu": nn.SiLU,
	"sigmoid": nn.Sigmoid,
	"identity": None,
	"none": None,
}

def build_activation_fn(cfg: dict[str, Any]) -> nn.Module | None:
    """Return a torch activation class for a simple string name."""
    name = str(cfg.pop("name")).lower()
    if name not in ACTIVATION_CLASSES:
        available = ", ".join(sorted(ACTIVATION_CLASSES.keys()))
        raise ValueError(f"Unsupported activation '{name}'. Available: {available}")
    ACTIVATION_CLASS = ACTIVATION_CLASSES[name]
    return ACTIVATION_CLASS(**cfg) if ACTIVATION_CLASS is not None else None

def build_optimizer(model: nn.Module, cfg: dict[str, Any]) -> Optimizer:
    """Build optimizer from config."""
    name = str(cfg.pop("name")).lower()
    match name:
        case "adam":
            return Adam(model.parameters(), **cfg)
        case "adamw":
            return AdamW(model.parameters(), **cfg)
        case "sgd":
            return SGD(model.parameters(), **cfg)
        case _:
            raise ValueError(f"Unsupported optimizer: {name}")


def build_loss(cfg: dict[str, Any]) -> nn.Module:
    """Build loss function from config."""
    name = str(cfg.get("supervised", {}).get("name", "mse")).lower()
    if name == "mse":
        return nn.MSELoss()
    if name == "mae":
        return nn.L1Loss()
    if name == "huber":
        delta = float(cfg.get("supervised", {}).get("huber_delta", 1.0))
        return nn.HuberLoss(delta=delta)
    raise ValueError(f"Unsupported loss: {name}")

def build_scheduler(optimizer: Optimizer, cfg: dict[str, Any], max_epochs: int) -> lr_scheduler.LRScheduler:
    """Build learning-rate scheduler from config"""
    name = str(cfg.pop('name')).lower()
    match name:
        case "step":
            return lr_scheduler.StepLR(optimizer, **cfg)
        case "cosine_annealing":
            return lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, **cfg)
        case "exp":
            return lr_scheduler.ExponentialLR(optimizer, **cfg)
        case _:
            raise ValueError(f"Unsupported scheduler: {name}")

def resolve_device(cfg: dict[str, Any], verbose: bool = True) -> torch.device:
    """Resolve device from config, fallback to CPU if CUDA unavailable."""
    requested = str(cfg.get("device", "cuda")).lower()
    if requested == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Request {requested} device - Using {device}")
    return device