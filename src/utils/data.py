"""Data loading utilities."""

from typing import Any, Union

import numpy as np, torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, TensorDataset

from src.dataset.core import FaultIdentificationDataset


def pick_key(arrays: dict[str, Any], candidates: list[str], role: str) -> np.ndarray:
    """Extract array from dict using fallback key names."""
    for key in candidates:
        if key in arrays:
            return arrays[key]
    raise KeyError(f"Could not find '{role}' in npz. Tried: {candidates}")


def load_npz_dataset(path: str) -> TensorDataset:
    """Load numpy npz file and convert to PyTorch TensorDataset."""
    arr = np.load(path)
    x = pick_key(arr, ["x", "X", "inputs", "features"], "inputs")
    y = pick_key(arr, ["y", "Y", "targets", "theta"], "targets")

    x_t = torch.as_tensor(x, dtype=torch.float32)
    y_t = torch.as_tensor(y, dtype=torch.float32)
    return TensorDataset(x_t, y_t)


def load_dataset(
    path: str,
    n_samples: int = 1,
    target_mode: str = "instant",
    ema_alpha: float = 0.1,
):
    """
    Auto-select dataset implementation based on npz keys.

    - Trajectory format (x, u, theta): use FaultIdentificationDataset
    - Flat x/y format: use TensorDataset wrapper
    """
    with np.load(path) as arr:
        keys = set(arr.files)

    if {"x", "u", "theta"}.issubset(keys):
        return FaultIdentificationDataset(
            path,
            n_samples=n_samples,
            target_mode=target_mode,
            ema_alpha=ema_alpha,
        )

    return load_npz_dataset(path)


def load_datasets(
    paths: Union[str, list[str]],
    n_samples: int = 1,
    target_mode: str = "instant",
    ema_alpha: float = 0.1,
) -> Dataset:
    """Load one or more dataset paths and return a single Dataset."""
    if isinstance(paths, str):
        paths = [paths]
    datasets = []
    N = len(paths)
    for i, p in enumerate(paths):
        datasets.append(
            load_dataset(
                p,
                n_samples=n_samples,
                target_mode=target_mode,
                ema_alpha=ema_alpha,
            )
        )
        print(f"Successfully loaded dataset {i+1}/{N} - {p}")
    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)


def make_loader(cfg: dict[str, Any], split: str) -> DataLoader:
    """Create a DataLoader for a given split: train | val | test."""
    data_cfg = cfg["dataset"]
    batch_cfg = data_cfg["batch"]
    load_cfg = data_cfg.get("loading", {})

    path_key = f"{split}_path"
    if path_key not in data_cfg:
        raise KeyError(f"Missing datasets path key: {path_key}")

    batch_key = f"{split}_batch_size"
    default_batch = 256 if split == "train" else 1024

    features_cfg = data_cfg.get("features", {})
    n_samples = int(features_cfg.get("n_samples", 1))
    target_mode = str(features_cfg.get("target_mode", "instant"))
    ema_alpha = float(features_cfg.get("ema_alpha", 0.1))
    dataset = load_datasets(
        data_cfg[path_key],
        n_samples=n_samples,
        target_mode=target_mode,
        ema_alpha=ema_alpha,
    )
    num_workers = int(batch_cfg.get("num_workers", 0))
    requested_device = str(cfg.get("device", "cuda")).lower()
    pin_memory = (requested_device == "cuda" and torch.cuda.is_available())
    persistent_workers = bool(batch_cfg.get("persistent_workers", False)) and num_workers > 0
    prefetch_factor = int(batch_cfg.get("prefetch_factor", 2)) if num_workers > 0 else None
    return DataLoader(
        dataset,
        batch_size=int(batch_cfg.get(batch_key, default_batch)),
        shuffle=bool(load_cfg.get(f"shuffle_{split}", split == "train")),
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        drop_last=bool(load_cfg.get(f"drop_last_{split}", split == "train")),
    )
