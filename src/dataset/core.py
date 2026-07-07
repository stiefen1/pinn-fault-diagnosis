from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class FaultIdentificationDataset(Dataset):
    """Per-step supervised dataset: [x(k), u(k), x(k+1)] -> theta(k)."""

    def __init__(self, src: str):
        arr = np.load(src)

        # Episode trajectories from generator:
        # x: (E, T+1, nx), u: (E, T, nu), theta: (E, T, ntheta)
        x = np.asarray(arr["x"], dtype=np.float32)
        u = np.asarray(arr["u"], dtype=np.float32)
        theta = np.asarray(arr["theta"], dtype=np.float32)

        if "lengths" in arr:
            lengths = np.asarray(arr["lengths"], dtype=np.int64)
        else:
            lengths = np.full((x.shape[0],), u.shape[1], dtype=np.int64)

        xk_list = []
        uk_list = []
        xkp1_list = []
        y_list = []

        for ep in range(x.shape[0]):
            t = int(lengths[ep])
            if t <= 0:
                continue

            xk_list.append(x[ep, :t, :])
            uk_list.append(u[ep, :t, :])
            xkp1_list.append(x[ep, 1 : t + 1, :])
            y_list.append(theta[ep, :t, :])

        if len(y_list) == 0:
            self.x = torch.zeros((0, x.shape[-1] + u.shape[-1] + x.shape[-1]), dtype=torch.float32)
            self.y = torch.zeros((0, theta.shape[-1]), dtype=torch.float32)
            self.nx = int(x.shape[-1])
            self.nu = int(u.shape[-1])
            self.ntheta = int(theta.shape[-1])
            return

        xk_flat = np.concatenate(xk_list, axis=0)
        uk_flat = np.concatenate(uk_list, axis=0)
        xkp1_flat = np.concatenate(xkp1_list, axis=0)
        y_flat = np.concatenate(y_list, axis=0)

        features = np.concatenate([xk_flat, uk_flat, xkp1_flat], axis=1)

        self.x = torch.as_tensor(features, dtype=torch.float32)
        self.y = torch.as_tensor(y_flat, dtype=torch.float32)
        self.nx = int(x.shape[-1])
        self.nu = int(u.shape[-1])
        self.ntheta = int(theta.shape[-1])

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]
    
if __name__ == "__main__":
    import os
    dataset = FaultIdentificationDataset(os.path.join('data', 'generated', 'fault_identification_dataset_v1.npz'))
    print(len(dataset), dataset)
    print(dataset[0][0].shape, dataset[0][1].shape)