import torch, torch.nn as nn
from torch import Tensor
from src.architecture.base import BaseFaultEstimatorNN, register_model
from src.utils.builders import build_activation_fn
class MLPFaultEstimator(BaseFaultEstimatorNN):
	def init_architecture(self) -> None:
		hidden_layers = list(self.architecture_cfg["hidden_layers"])
		if not hidden_layers:
			hidden_layers = [20, 10]

		activation_fn = build_activation_fn(self.architecture_cfg["activation"])
		output_activation_fn = build_activation_fn(
			self.architecture_cfg["output_activation"]
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
	model = MLPFaultEstimator(12, 8).to(device)
	x0 = torch.rand(1, 2 * 12 + 8)
	y = model.forward(x0)
	print(f"Output: {y} with shape {y.shape}")
