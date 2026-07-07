import torch
from torch import nn
from torch.utils.data import DataLoader

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
	model.eval()
	total_loss = 0.0
	count = 0
	for x, y in loader:
		x = x.to(device, non_blocking=True)
		y = y.to(device, non_blocking=True)
		pred = model(x)
		loss = criterion(pred, y)
		bs = x.shape[0]
		total_loss += float(loss.item()) * bs
		count += bs
	return total_loss / max(1, count)