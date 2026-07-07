"""Utilities for RNG control and simple sampling helpers."""

import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False, benchmark: bool = True) -> None:
    """Set random seeds and backend reproducibility/performance flags."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic mode prioritizes reproducibility over speed.
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = deterministic

    if deterministic:
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = benchmark


def sample_uniform_min_max(rng: np.random.Generator, spec: dict) -> float:
    """Sample uniformly from [min, max] in a mapping with min/max keys."""
    lo = float(spec["min"])
    hi = float(spec["max"])
    return float(rng.uniform(lo, hi))


def sample_clipped_value(rng: np.random.Generator, spec: dict) -> float:
    """
    Sample a value from a config-like mapping and clip to [min, max].

    Uses clipped Gaussian if both mean and std are provided, otherwise uniform.
    """
    lo = float(spec["min"])
    hi = float(spec["max"])
    if "mean" in spec and "std" in spec:
        sampled = rng.normal(float(spec["mean"]), float(spec["std"]))
    else:
        sampled = rng.uniform(lo, hi)

    return float(np.clip(sampled, lo, hi))
