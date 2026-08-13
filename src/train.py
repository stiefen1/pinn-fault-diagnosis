import argparse
import json
import os
from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.architecture.base import create_model
import src.architecture.mlp  # noqa: F401  # Registers MLPFaultEstimator and FaultEstimatorNN alias
from src.utils.builders import build_loss, build_optimizer, resolve_device, build_scheduler
from src.utils.configs import load_config, apply_overrides
from src.utils.data import make_loader
from src.utils.random import set_seed

from src.eval.core import evaluate

def save_checkpoint(
	path: Path,
	epoch: int,
	model: nn.Module,
	optimizer: Optimizer,
	val_loss: float,
	cfg: dict,
) -> None:
	torch.save(
		{
			"epoch": epoch,
			"model_state_dict": model.state_dict(),
			"optimizer_state_dict": optimizer.state_dict(),
			"val_loss": val_loss,
			"config": cfg,
		},
		path,
	)

def prune_periodic_checkpoints(ckpt_dir: Path, keep_last_k: int) -> None:
	"""
	Remove old checkpoint folders
	"""
	periodic = sorted(ckpt_dir.glob("epoch_*.pt"))
	if keep_last_k < 0:
		return
	if keep_last_k == 0:
		for path in periodic:
			path.unlink(missing_ok=True)
		return
	if len(periodic) <= keep_last_k:
		return

	for path in periodic[:-keep_last_k]:
		path.unlink(missing_ok=True)


def train_one_epoch(
	model: nn.Module,
	loader: DataLoader,
	optimizer: Optimizer,
	criterion: nn.Module,
	device: torch.device,
	epoch: int,
	log_every_n_steps: int,
	writer: SummaryWriter | None,
	global_step: int,
) -> tuple[float, int]:
	
	# model was already sent to device earlier
	model.train() # Set module in training mode (e.g. batch norm, dropout)
	total_loss = 0.0
	count = 0

	for step_idx, (x, y) in enumerate(loader, start=1):
		x = x.to(device, non_blocking=True)
		y = y.to(device, non_blocking=True)
		optimizer.zero_grad(set_to_none=True)

		# Forward pass
		pred = model(x)
		loss = criterion(pred, y)

		# Optimization step
		loss.backward()
		optimizer.step()

		bs = x.shape[0]
		total_loss += float(loss.detach().item()) * bs
		count += bs
		global_step += 1

		if log_every_n_steps > 0 and step_idx % log_every_n_steps == 0:
			step_loss = float(loss.detach().item())
			if writer is not None:
				writer.add_scalar("loss/train_step", step_loss, global_step)
			print(
				f"epoch={epoch} step={step_idx}/{len(loader)} " 
				f"train_step_loss={step_loss:.6f}"
			)

	return total_loss / max(1, count), global_step

def train(cfg: dict) -> float:
	seed_cfg = cfg["seed"]
	seed = int(seed_cfg["global"])
	deterministic = bool(seed_cfg["deterministic"])
	benchmark = bool(seed_cfg["torch"]["benchmark"])
	set_seed(seed, deterministic=deterministic, benchmark=benchmark)

	device = resolve_device(cfg, verbose=True)
	project_cfg = cfg["project"]
	io_cfg = cfg["io"]

	output_root = Path(project_cfg["output_root"])
	experiment_name = str(project_cfg["experiment_name"])
	run_dir = output_root / experiment_name
	ckpt_dir = run_dir / str(io_cfg["checkpoint_dir"])
	ckpt_dir.mkdir(parents=True, exist_ok=True)

	log_cfg = io_cfg["logging"]
	log_every_n_steps = int(io_cfg["log_every_n_steps"])
	save_every_n_epochs = int(io_cfg["save_every_n_epochs"])
	keep_last_k = int(io_cfg["keep_last_k"])
	writer = None
	if bool(log_cfg["enabled"]):
		log_dir = Path(str(log_cfg["log_dir"]))
		flush_secs = int(log_cfg["flush_secs"])
		writer = SummaryWriter(
			log_dir=str(log_dir),
			flush_secs=flush_secs,
		)

	model = create_model(cfg).to(device)

	train_cfg = cfg["train"]
	max_epochs = int(train_cfg["max_epochs"])
	early_stopping_cfg = train_cfg["early_stopping"]
	early_stopping_enabled = bool(early_stopping_cfg["enabled"])
	patience = int(early_stopping_cfg["patience"])
	min_delta = float(early_stopping_cfg["min_delta"])

	optimizer_cfg = cfg["optimizer"]
	scheduler_cfg = optimizer_cfg.pop("scheduler")
	optimizer = build_optimizer(model, optimizer_cfg)

	# Setup scheduler
	if scheduler_cfg.pop("enabled"): 
		scheduler = build_scheduler(optimizer, scheduler_cfg, max_epochs)
	else: 
		scheduler = None

	criterion = build_loss(cfg["loss"])
	train_loader = make_loader(cfg, "train")
	val_loader = make_loader(cfg, "val")

	best_val = float("inf")
	bad_epochs = 0
	global_step = 0

	for epoch in range(1, max_epochs + 1):
		train_loss, global_step = train_one_epoch(
			model=model,
			loader=train_loader,
			optimizer=optimizer,
			criterion=criterion,
			device=device,
			epoch=epoch,
			log_every_n_steps=log_every_n_steps,
			writer=writer,
			global_step=global_step,
		)

		val_loss = evaluate(model, val_loader, criterion, device)

		if writer is not None:
			writer.add_scalar("loss/train", train_loss, epoch)
			writer.add_scalar("loss/val", val_loss, epoch)

		print(f"epoch={epoch}/{max_epochs} | train_loss ({cfg['loss']['supervised']['name']})={train_loss:.6f} | val_loss={val_loss:.6f} | lr={optimizer.param_groups[0]['lr']:.3e}")

		improved = val_loss < (best_val - min_delta)
		if improved:
			best_val = val_loss
			bad_epochs = 0
			save_checkpoint(
				path=ckpt_dir / "best.pt",
				epoch=epoch,
				model=model,
				optimizer=optimizer,
				val_loss=val_loss,
				cfg=cfg,
			)
		else:
			bad_epochs += 1

		save_checkpoint(
			path=ckpt_dir / "last.pt",
			epoch=epoch,
			model=model,
			optimizer=optimizer,
			val_loss=val_loss,
			cfg=cfg,
		)

		if save_every_n_epochs > 0 and epoch % save_every_n_epochs == 0:
			save_checkpoint(
				path=ckpt_dir / f"epoch_{epoch:04d}.pt",
				epoch=epoch,
				model=model,
				optimizer=optimizer,
				val_loss=val_loss,
				cfg=cfg,
			)
			prune_periodic_checkpoints(ckpt_dir, keep_last_k)

		if early_stopping_enabled and bad_epochs >= patience:
			print(f"Early stopping at epoch {epoch}.")
			break

		if scheduler is not None:
			scheduler.step()

	if writer is not None:
		writer.close()

	return best_val


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--config", "-c", type=str, default="configs/train.yaml")
	parser.add_argument(
		"--set",
		nargs="*",
		default=[],
		metavar="key.path=value",
		help="Override config values, e.g. --set optimizer.lr=1e-4 train.max_epochs=100",
	)
	parser.add_argument(
		"--combinations",
		type=str,
		default=None,
		metavar="PATH",
		help="JSON file of hyperparameter combinations for SLURM array jobs. "
			 "The row at $SLURM_ARRAY_TASK_ID is applied as config overrides.",
	)
	args = parser.parse_args()

	cfg = load_config(Path(args.config).resolve())
	if args.set:
		cfg = apply_overrides(cfg, args.set)
	if args.combinations:
		task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
		combos = json.loads(Path(args.combinations).read_text())
		combo = combos[task_id]
		overrides = [f"{k}={v}" for k, v in combo.items()]
		cfg = apply_overrides(cfg, overrides)
		print(f"[SLURM array task {task_id}] Applying overrides: {overrides}")

		# Give each array task its own subdirectory so runs don't clobber each
		# other's checkpoints and TensorBoard can distinguish them.
		task_slug = f"task_{task_id:04d}"
		cfg["project"]["experiment_name"] = (
			str(cfg["project"]["experiment_name"]) + "/" + task_slug
		)
		cfg["io"]["logging"]["log_dir"] = (
			str(cfg["io"]["logging"]["log_dir"]) + "/" + task_slug
		)
	train(cfg)


if __name__ == "__main__":
	main()