"""Checkpoint path utilities."""

from pathlib import Path
from typing import Any


def resolve_run_dir(cfg: dict[str, Any]) -> Path:
    project_cfg = cfg.get("project", {})
    output_root = Path(str(project_cfg.get("output_root", "runs")))
    experiment_name = str(project_cfg.get("experiment_name", "experiment"))
    return output_root / experiment_name


def resolve_checkpoint_dir(cfg: dict[str, Any]) -> Path:
    io_cfg = cfg.get("io", {})
    return resolve_run_dir(cfg) / str(io_cfg.get("checkpoint_dir", "checkpoints"))


def resolve_test_checkpoint_path(cfg: dict[str, Any]) -> Path:
    test_cfg = cfg.get("test", {})
    checkpoint = str(test_cfg.get("checkpoint", "best")).lower()

    if checkpoint == "path":
        cp = test_cfg.get("checkpoint_path")
        if cp is None:
            raise ValueError("test.checkpoint is 'path' but test.checkpoint_path is null")
        return Path(str(cp))

    ckpt_dir = resolve_checkpoint_dir(cfg)
    if checkpoint == "last":
        return ckpt_dir / "last.pt"
    return ckpt_dir / "best.pt"
