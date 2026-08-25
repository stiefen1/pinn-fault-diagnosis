import torch, torch.nn as nn, numpy as np
from torch import Tensor
from src.architecture.base import LearningBasedFaultEstimator, register_model
from src.utils.builders import build_activation_fn

from python_vehicle_simulator.lib.weather import Wind, Current

DEFAULT_ACTIVATION_CFG = {
	"name": "relu"
}

class CNNFaultEstimator(LearningBasedFaultEstimator):
	def init_architecture(self) -> None:
		raise NotImplementedError(f"")

	def forward(self, x: Tensor) -> Tensor:
		raise NotImplementedError(f"")

register_model("CNNFaultEstimator", CNNFaultEstimator)

if __name__ == "__main__":
	device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"  # type: ignore
	print(f"Using {device} device")
	model = CNNFaultEstimator(9, 6, n_samples=3).to(device)
	x0 = torch.rand(1, 3, 2 * 9 + 6 + 2 + 2) # 2 + 2 = wind, current
	y = model.forward(x0)
	print(f"Output: {y} with shape {y.shape}")