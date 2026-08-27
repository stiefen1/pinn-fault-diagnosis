"""Utilities for RNG control and simple sampling helpers."""

import random

import numpy as np
import torch

from typing import Any


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

def sample_clipped_value(rng: np.random.Generator, spec: dict) -> float | list[float]:
    """
    Sample a value from a config-like mapping and clip to [min, max].

    Uses clipped Gaussian if both mean and std are provided, otherwise uniform.
    """
    if isinstance(spec["min"], list):
        lo = np.array([float(str(v)) for v in spec["min"]])
    else:
        lo = float(spec["min"])
    if isinstance(spec["max"], list):
        hi = np.array([float(str(v)) for v in spec["max"]])
    else:
        hi = float(spec["max"])

    if "mean" in spec and "std" in spec:
        if isinstance(spec["mean"], list):
            mean, std = np.array([float(str(v)) for v in spec["mean"]]), np.array([float(str(v)) for v in spec["std"]])
        else:
            mean, std = float(spec["mean"]), float(spec["std"])
        sampled = rng.normal(mean, std)
    else:
        sampled = rng.uniform(lo, hi)

    return float(np.clip(sampled, lo, hi)) if isinstance(sampled, float) else sampled.tolist()

def build_fault_schedule(
    faults_cfg: dict[str, Any],
    max_steps: int,
    healthy_value: float,
) -> np.ndarray:
    schedule = np.full((max_steps, 6), healthy_value, dtype=np.float32) # n_steps x ntheta -> fault parameter along time vector

    azimuth_cfg = faults_cfg["azimuth_stucked"] # fault amplitude \in [0, 1]
    loe_cfg = faults_cfg["loss_of_effectiveness"] # fault amplitude \in [0, 1]

    for i in range(3):
        if azimuth_cfg[i] is not None:
            k_fault = int(np.floor(azimuth_cfg[i]["t_norm"] * (max_steps - 1))) # map normalized time to time-step
            schedule[k_fault:, i] = azimuth_cfg[i]["amp"] # theta only changes once along a trajectory
    for i in range(3):
        if loe_cfg[i] is not None:
            k_fault = int(np.floor(loe_cfg[i]["t_norm"] * (max_steps - 1))) # map normalized time to time-step
            schedule[k_fault:, i+3] = loe_cfg[i]["amp"] # theta only changes once along a trajectory
    return schedule

def sample_fault_cfg(rng: np.random.Generator, faults_cfg: dict) -> list[dict | None]:
    amp_cfg = faults_cfg["amplitude"] # fault amplitude \in [0, 1]
    time_cfg = faults_cfg["time"]
    is_integer = faults_cfg.get("is_integer", False)
    probs = np.asarray(faults_cfg["prob"], dtype=float) # tuple[p1, p2, p3]
    cfg = []

    # We always have multiple faults at the same time, which makes the task VERY challenging.
    # We should focus first on single faults
    possible_fault_idx = np.arange(probs.shape[0]).tolist()
    w_tot = np.sum(probs).astype(float)
    if faults_cfg["single"] and w_tot > 0.0:
        possible_fault_idx = [rng.choice(len(probs), p=probs/w_tot)]

    for i in range(probs.shape[0]):
        if i in possible_fault_idx and rng.random() < probs[i]:
            t_norm = sample_clipped_value(rng, time_cfg) # randomy sample fault time, normalized \in [0, 1] i.e equals 1 if t_fault = tf
            assert isinstance(t_norm, float), f"normalized fault time must be float, got t_norm={t_norm} of type {type(t_norm)}"

            amp = sample_clipped_value(rng, amp_cfg) # randomly sample fault amplitude
            assert isinstance(amp, float), f"fault amplitude must be float, got amp={amp} of type {type(amp)}"
            if is_integer:
                amp = int(amp)

            cfg.append({"t_norm": t_norm, "amp": amp})
        else:
            cfg.append(None)
    return cfg

    

    # azimuth_sched = build_fault_schedule(
    #     rng,
    #     faults_cfg["azimuth_stucked"],
    #     max_steps=max_steps,
    #     healthy_value=1.0,
    # )
    # loe_sched = build_fault_schedule(
    #     rng,
    #     faults_cfg["loss_of_effectiveness"],
    #     max_steps=max_steps,
    #     healthy_value=1.0,
    # )

    # theta_schedule = np.concatenate([azimuth_sched, loe_sched], axis=1)
    # return theta_schedule.astype(np.float32)


