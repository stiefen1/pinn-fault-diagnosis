from abc import ABC, abstractmethod
from typing import Any

import torch.nn as nn
from torch import Tensor, load

from src.utils.builders import ACTIVATION_CLASSES

class LearningBasedFaultEstimator(nn.Module, ABC):
	architecture: nn.Sequential

	def __init__(
		self,
		ny: int,
		nu: int,
		n_samples: int = 1,
		ntheta: int = 3,
		architecture_cfg: dict[str, Any] | None = None,
	):
		super().__init__()
		self.ny = ny
		self.nu = nu
		self.n_samples = n_samples
		self.ntheta = ntheta
		self.architecture_cfg = architecture_cfg or {}
		self.init_architecture()
		self.init_weights() # unused for now

	@property
	def input_dim(self) -> int:
		return self.n_samples * (2 * self.ny + self.nu + 2 + 2) # 2 + 2 = wind + current

	@property
	def output_dim(self) -> int:
		return self.ntheta

	@abstractmethod
	def init_architecture(self) -> None:
		pass
	
	@abstractmethod
	def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
		pass    

	def init_weights(self) -> None:
		pass
		# modules = list(self.architecture) # contains both layers and activation fn
		# n_modules = len(modules)

		# for i in range(n_modules):
		# 	module = modules[i]
		# 	use e.g. nn.init.xavier_normal_(module.weight) to initialize parameters of a layer
			
			

MODEL_REGISTRY: dict[str, type[LearningBasedFaultEstimator]] = {}


def register_model(name: str, model_cls: type[LearningBasedFaultEstimator]) -> None:
	MODEL_REGISTRY[name] = model_cls


def create_model(cfg: dict[str, Any]) -> LearningBasedFaultEstimator:
	model_cfg = cfg["model"]
	features_cfg = cfg["dataset"]["features"]

	model_name = str(model_cfg["name"])
	if model_name not in MODEL_REGISTRY:
		available = ", ".join(sorted(MODEL_REGISTRY.keys()))
		raise ValueError(f"Unknown model '{model_name}'. Available: {available}")

	model_cls = MODEL_REGISTRY[model_name]
	model = model_cls(
		ny=int(features_cfg["ny"]),
		nu=int(features_cfg["nu"]),
		n_samples=int(features_cfg["n_samples"]),
		ntheta=int(features_cfg["ntheta"]),
		architecture_cfg=model_cfg["architecture"],
	)
	resume_from = cfg["io"].get("resume_from", None)
	if resume_from is not None:
		ckpt = load(resume_from, map_location="cpu", weights_only=True)
		model.load_state_dict(ckpt["model_state_dict"])
		print(f"Loaded weights from: {resume_from}")
	return model


if __name__ == "__main__":
	import argparse, sys, torch
	from pathlib import Path
	
	# Import models BEFORE anything else to populate registry
	import src.architecture.mlp  # noqa
	from src.utils.configs import load_config
	
	# Use the module's registry (which got populated), not the local one
	base_module = sys.modules["src.architecture.base"]
	registry = base_module.MODEL_REGISTRY
	print(f"Registry: {registry}")
	
	parser = argparse.ArgumentParser(description="Sanity check: load config, instantiate model, forward pass")
	parser.add_argument("--config", "-c", default="configs/base.yaml", help="Config file path (default: configs/base.yaml)")
	args = parser.parse_args()
	
	cfg = load_config(Path(args.config))
	device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"  # type: ignore
	print(f"Using {device} device")
	
	model_cfg = cfg["model"]
	features_cfg = cfg["data"]["features"]
	model_name = str(model_cfg["name"])
	if model_name not in registry:
		available = ", ".join(sorted(registry.keys()))
		raise ValueError(f"Unknown model '{model_name}'. Available: {available}")
	
	model_cls = registry[model_name]
	model = model_cls(
		ny=int(features_cfg["ny"]),
		nu=int(features_cfg["nu"]),
		n_samples=int(features_cfg["n_samples"]),
		ntheta=int(features_cfg["ntheta"]),
		architecture_cfg=model_cfg["architecture"],
	).to(device)
	print(f"Model: {model.__class__.__name__}")
	
	ny, nu = int(features_cfg["ny"]), int(features_cfg["nu"])
	n_samples = int(features_cfg["n_samples"])
	input_dim = n_samples * (2 * ny + nu)
	
	x = torch.rand(1, input_dim).to(device)
	y = model.forward(x)
	print(f"Output: {y} with shape {y.shape}")
