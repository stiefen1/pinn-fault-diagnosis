"""Configuration loading and merging utilities."""

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    """Load YAML file into a dictionary."""
    with path.open("r", encoding="utf-8") as f:
        content = yaml.safe_load(f) or {}
    if not isinstance(content, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return content


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base. Override values take priority."""
    merged = deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: Path) -> dict[str, Any]:
    """Load overlay config and merge with base if referenced."""
    overlay = load_yaml(config_path)
    base_ref = overlay.get("config", {}).get("base")
    if not base_ref:
        return overlay

    base_ref_path = Path(str(base_ref))
    if base_ref_path.is_absolute():
        base_path = base_ref_path
    else:
        candidate_overlay_relative = (config_path.parent / base_ref_path).resolve()
        candidate_cwd_relative = (Path.cwd() / base_ref_path).resolve()

        if candidate_overlay_relative.exists():
            base_path = candidate_overlay_relative
        elif candidate_cwd_relative.exists():
            base_path = candidate_cwd_relative
        else:
            base_path = candidate_overlay_relative

    base = load_yaml(base_path)
    return deep_merge(base, overlay)


def apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply dot-notation overrides to a config dict.

    Each override is a string of the form ``"section.key=value"``.
    Values are parsed by ``yaml.safe_load``, so ints, floats, bools,
    lists, and strings all work as expected:

        apply_overrides(cfg, [
            "optimizer.lr=1e-4",
            "model.architecture.hidden_layers=[256,128,64]",
            "train.early_stopping.enabled=true",
        ])

    This enables to run python files and override parameters from the configuration file
    """
    cfg = deepcopy(cfg)
    for override in overrides:
        key_path, _, raw_value = override.partition("=")
        if not _:
            raise ValueError(f"Override must be 'key.path=value', got: {override!r}")
        keys = key_path.strip().split(".")
        value = yaml.safe_load(raw_value)
        node = cfg
        for k in keys[:-1]:
            if k not in node or not isinstance(node[k], dict):
                node[k] = {}
            node = node[k]
        node[keys[-1]] = value
    return cfg
