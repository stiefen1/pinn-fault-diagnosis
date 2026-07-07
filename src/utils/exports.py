"""Export helpers for evaluation outputs."""

from typing import Any
from pathlib import Path

import numpy as np

from src.utils.checkpoints import resolve_run_dir


def maybe_export_test_predictions(cfg: dict[str, Any], preds: np.ndarray, targets: np.ndarray) -> None:
    """Export test predictions depending on test.export config flags."""
    test_cfg = cfg.get("test", {})
    export_cfg = test_cfg.get("export", {})
    if not isinstance(export_cfg, dict):
        return

    run_dir = resolve_run_dir(cfg)
    output_dir = run_dir / "test_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    if bool(export_cfg.get("save_npz", False)):
        np.savez(output_dir / "predictions.npz", pred=preds, target=targets)

    if bool(export_cfg.get("save_csv", False)):
        pred_flat = preds.reshape(preds.shape[0], -1)
        target_flat = targets.reshape(targets.shape[0], -1)
        headers = [
            *[f"pred_{i}" for i in range(pred_flat.shape[1])],
            *[f"target_{i}" for i in range(target_flat.shape[1])],
        ]
        stacked = np.concatenate([pred_flat, target_flat], axis=1)
        np.savetxt(
            output_dir / "predictions.csv",
            stacked,
            delimiter=",",
            header=",".join(headers),
            comments="",
        )
