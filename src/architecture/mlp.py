import torch, torch.nn as nn, numpy as np
from torch import Tensor
from src.architecture.base import LearningBasedFaultEstimator, register_model
from src.utils.builders import build_activation_fn

from python_vehicle_simulator.lib.weather import Wind, Current

DEFAULT_ACTIVATION_CFG = {
	"name": "relu"
}

class MLPFaultEstimator(LearningBasedFaultEstimator):
	def init_architecture(self) -> None:
		hidden_layers = self.architecture_cfg.get("hidden_layers", [20, 10])

		activation_fn = build_activation_fn(
			self.architecture_cfg.get("activation", DEFAULT_ACTIVATION_CFG)
		)
		output_activation_fn = build_activation_fn(
			self.architecture_cfg.get("output_activation", DEFAULT_ACTIVATION_CFG)
		)

		self.flatten = nn.Flatten()
		layers: list[nn.Module] = []
		in_features = self.input_dim

		for out_features in hidden_layers:
			layers.append(nn.Linear(in_features=in_features, out_features=int(out_features)))
			if activation_fn is not None:
				layers.append(activation_fn)
			in_features = int(out_features)

		layers.append(nn.Linear(in_features=in_features, out_features=self.ntheta))
		if output_activation_fn is not None:
			layers.append(output_activation_fn)

		self.architecture = nn.Sequential(*layers)

	def forward(self, x: Tensor) -> Tensor:
		x = self.flatten(x)
		return self.architecture(x)

register_model("MLPFaultEstimator", MLPFaultEstimator)

if __name__ == "__main__":
	device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"  # type: ignore
	print(f"Using {device} device")
	model = MLPFaultEstimator(9, 6, n_samples=3).to(device)
	x0 = torch.rand(1, 3, 2 * 9 + 6 + 2 + 2) # 2 + 2 = wind, current
	y = model.forward(x0)
	print(f"Output: {y} with shape {y.shape}")
