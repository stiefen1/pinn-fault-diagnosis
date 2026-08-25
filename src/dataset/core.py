from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class FaultIdentificationDataset(Dataset):
    """Per-step supervised dataset with instantaneous or EMA fault targets."""

    def __init__(
        self,
        src: str,
        n_samples: int = 1,
        target_mode: str = "instant",
        ema_alpha: float = 0.1,
    ):
        if target_mode not in {"instant", "ema"}:
            raise ValueError("target_mode must be 'instant' or 'ema'")
        if not 0.0 < ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1]")

        arr = np.load(src)

        # Episode trajectories from generator:
        # x: (E, T+1, ny), u: (E, T, nu), theta: (E, T, ntheta)
        x_hat = np.asarray(arr["x_hat"], dtype=np.float32) # We don't use x_hat anymore to avoid bad state estimation due to coupling
        y = np.asarray(arr["y"])
        u = np.asarray(arr["u"], dtype=np.float32)
        theta = np.asarray(arr["theta"], dtype=np.float32)
        wind_meas = np.asarray(arr["wind_meas"], dtype=np.float32)
        current_meas = np.asarray(arr["current_meas"], dtype=np.float32)

        if "lengths" in arr:
            lengths = np.asarray(arr["lengths"], dtype=np.int64)
        else:
            lengths = np.full((x_hat.shape[0],), u.shape[1], dtype=np.int64)

        self.ny = int(y.shape[-1])
        self.nx = int(x_hat.shape[-1])
        self.nu = int(u.shape[-1])
        self.ntheta = int(theta.shape[-1])
        step_dim = self.ny + self.nu + self.ny + wind_meas.shape[-1] + current_meas.shape[-1]  ##### WE SHOULD PROBABLY EXPRESS WIND IN SHIP FRAME

        input_list = []
        output_list = []

        for ep in range(x_hat.shape[0]):
            t = int(lengths[ep]) - 1
            if t < n_samples:
                continue

            # Per-step feature: [y_k, u_k, y_{k+1}, wind_k, current_k]
            step_features = np.concatenate([
                np.asarray(y[ep, :t, :], dtype=np.float32),
                u[ep, :t, :],
                np.asarray(y[ep, 1:t+1, :], dtype=np.float32),
                wind_meas[ep, :t, :],
                current_meas[ep, :t, :],
            ], axis=1)  # (t, step_dim)

            # Sliding window of n_samples consecutive steps; label at the last step
            n_windows = t - n_samples + 1
            windows = np.stack([
                step_features[i:i + n_samples].reshape(-1)
                for i in range(n_windows)
            ])  # (n_windows, n_samples * step_dim)
            if target_mode == "instant":
                targets = theta[ep, :t, :]
            else:
                targets = np.empty_like(theta[ep, :t, :])
                targets[0] = theta[ep, 0]
                for k in range(1, t):
                    targets[k] = (
                        ema_alpha * theta[ep, k]
                        + (1.0 - ema_alpha) * targets[k - 1]
                    )

            input_list.append(windows)
            output_list.append(targets[n_samples - 1:, :])

        if len(output_list) == 0:
            self.input = torch.zeros((0, n_samples * step_dim), dtype=torch.float32)
            self.output = torch.zeros((0, self.ntheta), dtype=torch.float32)
            return

        self.input = torch.as_tensor(np.concatenate(input_list, axis=0), dtype=torch.float32)
        self.output = torch.as_tensor(np.concatenate(output_list, axis=0), dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.input.shape[0])

    def __getitem__(self, idx: int):
        return self.input[idx], self.output[idx]
    
if __name__ == "__main__":
    import os
    dataset = FaultIdentificationDataset(os.path.join('data', 'test', 'fault_identification_dataset.npz'))
    print(len(dataset), dataset)
    print(dataset[0][0].shape, dataset[0][1].shape)