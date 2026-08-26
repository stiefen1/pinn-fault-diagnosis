import torch, torch.nn as nn
from torch import Tensor
from src.architecture.base import LearningBasedFaultEstimator, register_model
from src.utils.builders import build_activation_fn

DEFAULT_ACTIVATION_CFG = {
	"name": "relu"
}

class CNNFaultEstimator(LearningBasedFaultEstimator):
	@staticmethod
	def conv_output_dim(size: int, kernel_size: int, padding: int) -> int:
		return size + 2 * padding - kernel_size + 1

	def init_architecture(self) -> None:
		conv_channels = self.architecture_cfg.get("conv_channels", [16, 32])
		if len(conv_channels) != 2:
			raise ValueError("conv_channels must contain exactly two channel sizes")

		kernel_size = tuple(self.architecture_cfg.get("kernel_size", (3, 3)))
		padding = tuple(self.architecture_cfg.get("padding", (0, 0)))
		hidden_layers = self.architecture_cfg.get("hidden_layers", [128, 64])
		dropout = float(self.architecture_cfg.get("dropout", 0.0))

		activation_cfg = self.architecture_cfg.get("activation", DEFAULT_ACTIVATION_CFG)
		output_activation_cfg = self.architecture_cfg.get("output_activation", {"name": "identity"})

		self.features_per_step = self.input_dim // self.n_samples
		conv_height = self.n_samples
		conv_width = self.features_per_step
		for layer_kernel_size, layer_padding in zip(kernel_size, padding):
			conv_height = self.conv_output_dim(conv_height, int(layer_kernel_size), int(layer_padding))
			conv_width = self.conv_output_dim(conv_width, int(layer_kernel_size), int(layer_padding))
			if conv_height <= 0 or conv_width <= 0:
				raise ValueError("CNN kernel_size/padding make the convolution output empty")

		self.conv = nn.Sequential(
			nn.Conv2d(1, int(conv_channels[0]), kernel_size=int(kernel_size[0]), padding=padding[0]),
			build_activation_fn(activation_cfg) or nn.Identity(),
			nn.Conv2d(int(conv_channels[0]), int(conv_channels[1]), kernel_size=int(kernel_size[1]), padding=padding[1]),
			build_activation_fn(activation_cfg) or nn.Identity(),
		)

		layers: list[nn.Module] = []
		in_features = int(conv_channels[1]) * conv_height * conv_width
		for out_features in hidden_layers:
			layers.append(nn.Linear(in_features, int(out_features)))
			activation_fn = build_activation_fn(activation_cfg)
			if activation_fn is not None:
				layers.append(activation_fn)
			if dropout > 0.0:
				layers.append(nn.Dropout(dropout))
			in_features = int(out_features)

		layers.append(nn.Linear(in_features, self.ntheta))
		output_activation_fn = build_activation_fn(output_activation_cfg)
		if output_activation_fn is not None:
			layers.append(output_activation_fn)

		self.mlp = nn.Sequential(*layers)
		self.architecture = nn.Sequential(self.conv, nn.Flatten(), self.mlp)

	def forward(self, x: Tensor) -> Tensor:
		x = x.view(x.shape[0], self.n_samples, self.features_per_step)
		x = x.unsqueeze(1)
		x = self.conv(x)
		x = torch.flatten(x, start_dim=1)
		return self.mlp(x)

register_model("CNNFaultEstimator", CNNFaultEstimator)

if __name__ == "__main__":
	device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"  # type: ignore
	print(f"Using {device} device")
	model = CNNFaultEstimator(9, 6, n_samples=3).to(device)
	x0 = torch.rand(1, 3, 2 * 9 + 6 + 2 + 2) # 2 + 2 = wind, current
	y = model.forward(x0)
	print(f"Output: {y} with shape {y.shape}")