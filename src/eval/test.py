import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.architecture.base import create_model
import src.architecture.mlp  # noqa: F401  # Registers MLPFaultEstimator and FaultEstimatorNN alias
from src.utils.builders import build_loss, resolve_device
from src.utils.checkpoints import resolve_test_checkpoint_path
from src.utils.configs import load_config
from src.utils.data import make_loader
from src.utils.exports import maybe_export_test_predictions
from src.utils.random import set_seed


@torch.no_grad()
def evaluate_test_set(
	model: nn.Module,
	loader: DataLoader,
	criterion: nn.Module,
	device: torch.device,
) -> tuple[float, float, float, float, np.ndarray, np.ndarray]:
	model.eval()
	total_loss = 0.0
	total_mae = 0.0
	total_sqerr = 0.0
	count = 0

	preds_all: list[np.ndarray] = []
	targets_all: list[np.ndarray] = []

	for x, y in loader:
		x = x.to(device, non_blocking=True)
		y = y.to(device, non_blocking=True)

		pred = model(x)
		loss = criterion(pred, y)

		bs = x.shape[0]
		diff = pred - y
		total_loss += float(loss.item()) * bs
		total_mae += float(diff.abs().mean().item()) * bs
		total_sqerr += float(diff.pow(2).sum().item())
		count += bs

		preds_all.append(pred.detach().cpu().numpy())
		targets_all.append(y.detach().cpu().numpy())

	preds = np.concatenate(preds_all, axis=0)
	targets = np.concatenate(targets_all, axis=0)

	total_loss = total_loss / max(1, count)
	mae = total_mae / max(1, count)
	rmse = float(np.sqrt(total_sqerr / max(1, count * targets.shape[1])))

	target_mean = targets.mean(axis=0, keepdims=True)
	ss_res = float(((targets - preds) ** 2).sum())
	ss_tot = float(((targets - target_mean) ** 2).sum())
	r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

	return total_loss, rmse, mae, r2, preds, targets


def run_test(cfg: dict) -> None:
	seed_cfg = cfg.get("seed", {})
	seed = int(seed_cfg.get("global", 42))
	deterministic = bool(seed_cfg.get("deterministic", False))
	benchmark = bool(seed_cfg.get("torch", {}).get("benchmark", True))
	set_seed(seed, deterministic=deterministic, benchmark=benchmark)

	device = resolve_device(cfg)
	model = create_model(cfg).to(device)

	criterion = build_loss(cfg.get("loss", {}))
	test_loader = make_loader(cfg, "test")

	checkpoint_path = resolve_test_checkpoint_path(cfg)
	if not checkpoint_path.exists():
		raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

	ckpt = torch.load(checkpoint_path, map_location=device)
	model.load_state_dict(ckpt["model_state_dict"])

	test_loss, rmse, mae, r2, preds, targets = evaluate_test_set(
		model=model,
		loader=test_loader,
		criterion=criterion,
		device=device,
	)

	print(f"checkpoint={checkpoint_path}")
	print(f"test_loss={test_loss:.6f}")
	print(f"test_rmse={rmse:.6f}")
	print(f"test_mae={mae:.6f}")
	print(f"test_r2={r2:.6f}")

	maybe_export_test_predictions(cfg, preds, targets)


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--config", type=str, default="configs/test.yaml")
	args = parser.parse_args()

	cfg = load_config(Path(args.config).resolve())
	run_test(cfg)


if __name__ == "__main__":
	main()
