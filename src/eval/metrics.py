from abc import ABC, abstractmethod
from typing import Dict, Optional, List

import numpy as np, matplotlib.pyplot as plt
from matplotlib.axes import Axes

class IEvalMetric(ABC):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        pass

    @abstractmethod
    def calculate(self, data: np.ndarray, target: np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        Compute metric for a given pair of data and target values with shapes (N, M, K)
        """
        return {}

    def __call__(self, data: Dict[str, np.ndarray], target: Dict[str, np.ndarray], *args, **kwargs) -> Dict:
        """
        Input dictionnaries should contain numpy array of size (N, M, K):
        N episodes of length M with K dimensions

        Example:
        data["diagnosis_theta"].shape = (10, 500, 6) # 10 episodes with 500 timesteps representing 6 fault parameters
        """

        out = {}
        for key, val in data.items():
            assert len(val.shape) == 3, f"val must be 3D, got val.shape = {val.shape}"
            if not(key in target.keys()):
                if not(key.split('.')[0] in target.keys()):
                    print(f"{key} is a key of data, but was not found in target")
                continue
            
            # If target is provided as single zero, reshape it properly as an array of same size as data
            if isinstance(target[key], (int, float)):
                target[key] = np.ones_like(val) * target[key]

            # Builds dictionnary with same keys as data with calculated metric
            out[key] = self.calculate(val, target[key], *args, **kwargs)
        return out

class SE(IEvalMetric):
    """
    Squared Error as a function of time: e(t)**2
    """
    def __init__(
            self
        ):
        pass

    def calculate(self, data: np.ndarray, target: np.ndarray, **kwargs) -> np.ndarray:
        return np.pow(data - target, 2)

class MSE(SE):
    """
    Mean Squared Error
    """
    def __init__(
            self
        ):
        super().__init__()

    def calculate(self, data: np.ndarray, target: np.ndarray, **kwargs) -> np.ndarray:
        return np.mean(super().calculate(data, target, **kwargs), axis=1)

class RMSE(MSE):
    def __init__(
        self
    ):
        super().__init__()

    def calculate(self, data: np.ndarray, target: np.ndarray, **kwargs) -> np.ndarray:
        return np.sqrt(super().calculate(data, target, **kwargs)) 

class AE(IEvalMetric):
    """
    Absolute Error as a function of time |e(t)|
    """
    def __init__(
                self
            ):
        pass
    
    def calculate(self, data: np.ndarray, target: np.ndarray, **kwargs) -> np.ndarray:
        return np.abs(data - target)

class MAE(AE):
    """
    Mean Absolute Error
    """
    def __init__(
            self
        ):
        super().__init__()

    def calculate(self, data: np.ndarray, target: np.ndarray, **kwargs) -> np.ndarray:
        return np.mean(super().calculate(data, target, **kwargs), axis=1)

class IAE(AE):
    """
    Integral Average Error (IAE)
    """
    def __init__(
        self
    ):
        super().__init__()

    def calculate(self, data: np.ndarray, target: np.ndarray, **kwargs) -> np.ndarray:
        return np.cumsum(super().calculate(data, target, **kwargs), axis=1)

class ITAE(AE):
    """
    Integral Time Average Error (IAE) - Penalize slow fault estimation
    """
    def __init__(
        self
    ):
        super().__init__()

    def calculate(self, data: np.ndarray, target: np.ndarray, dt: float = 0.2, **kwargs) -> np.ndarray:
        t = np.cumsum(np.ones_like(data) * dt, axis=1) - dt # 0, dt, 2*dt, 3*dt, ..., (N-1)*dt
        return np.cumsum(t * super().calculate(data, target, **kwargs), axis=1)

class ITSE(SE):
    """
    Integral Time Squared Error (ITSE) - Penalize slow fault estimation
    """
    def __init__(
        self
    ):
        super().__init__()

    def calculate(self, data: np.ndarray, target: np.ndarray, dt: float = 0.2, **kwargs) -> np.ndarray:
        t = np.cumsum(np.ones_like(data) * dt, axis=1) - dt # 0, dt, 2*dt, 3*dt, ..., (N-1)*dt
        return np.cumsum(t * super().calculate(data, target, **kwargs), axis=1)

class Var(IEvalMetric):
    """
    Variance of the error signal
    """
    def __init__(
        self
    ):
        pass

    def calculate(self, data: np.ndarray, target: np.ndarray, **kwargs) -> np.ndarray:
        return np.var(data - target, axis=1)

class STD(IEvalMetric):
    """
    Standard deviation of the error signal
    """
    def __init__(
            self
        ):
            pass
    
    def calculate(self, data: np.ndarray, target: np.ndarray, **kwargs) -> np.ndarray:
        return np.std(data - target, axis=1)

class Evaluator:
    def __init__(
        self,
        keys: List[str],
    ):
        self.metrics: Dict[str, IEvalMetric] = {}
        for key in keys:
            match key.lower():
                case "se":
                    self.metrics.update({"SE": SE()})
                case "mse":
                    self.metrics.update({"MSE": MSE()})
                case "rmse":
                    self.metrics.update({"RMSE": RMSE()})
                case "ae":
                    self.metrics.update({"AE": AE()})
                case "mae":
                    self.metrics.update({"MAE": MAE()})
                case "iae":
                    self.metrics.update({"IAE": IAE()})
                case "itae":
                    self.metrics.update({"ITAE": ITAE()})
                case "itse":
                    self.metrics.update({"ITSE": ITSE()})
                case "std":
                    self.metrics.update({"STD": STD()})
                case "var": 
                    self.metrics.update({"VAR": Var()})

    def __call__(self, data: Dict, target: Dict, **kwargs) -> Dict:
        out = {}
        for key, metric in self.metrics.items():
            val = metric(data, target, **kwargs)
            out[key] = val
        return out

    def plot(self, data: Dict, target: Dict, dt: float = 1.0, **kwargs) -> List[plt.Axes]:
        # Expand target so dot-notation keys resolve: "theta.ekf" -> target["theta"]
        expanded_target = dict(target)
        for key in data:
            if key not in expanded_target:
                base = key.split(".")[0]
                if base in target:
                    expanded_target[key] = target[base]

        results = self(data, expanded_target, **kwargs)

        # Group data keys by base name: "theta.ekf", "theta.pf" -> {"theta": ["theta.ekf", "theta.pf"]}
        from collections import defaultdict
        base_groups: Dict[str, List[str]] = defaultdict(list)
        for key in data.keys():
            base_groups[key.split(".")[0]].append(key)

        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        all_axes = []

        for metric_name, res in results.items():
            for base, keys in base_groups.items():
                keys_in_res = [k for k in keys if k in res and len(res[k].shape) == 3]
                if not keys_in_res:
                    continue

                N, M, num_dims = res[keys_in_res[0]].shape
                time = np.arange(M) * dt

                fig, axes = plt.subplots(nrows=num_dims, ncols=1, figsize=(10, 2.5 * num_dims), sharex=True)
                if num_dims == 1:
                    axes = np.array([axes])

                for i, key in enumerate(keys_in_res):
                    val = res[key]
                    color = colors[i % len(colors)]
                    label = key.split(".", 1)[1] if "." in key else key
                    for k in range(num_dims):
                        for n in range(N):
                            axes[k].plot(time, val[n, :, k], color=color,
                                         label=label if n == 0 else "_nolegend_")
                        axes[k].set_ylabel(f"Dim {k+1} [-]")
                        axes[k].grid(True)

                if len(keys_in_res) > 1:
                    axes[0].legend()

                axes[-1].set_xlabel("Time (seconds)")
                fig.suptitle(f"{metric_name.upper()}({base.upper()})", fontsize=14, fontweight='bold')
                fig.tight_layout()
                all_axes.extend(axes)

        return all_axes

def test() -> None:
    import numpy as np

    # Fake dataset
    x = np.ones((10, 51, 3))
    xd = np.sin(np.linspace(0, 4*np.pi, num=51))[None, :, None] * np.ones((10, 51, 3))
    xd[:, :, 0] *= 2*np.sin(np.linspace(0, 10*np.pi, num=51))[None, :]
    xd[:, :, 1] = -4 * np.sin(np.linspace(0, 12*np.pi, num=51))[None, :]
    y = np.ones((10, 50, 2))
    yd = 0

    # Build input dictionnaries
    data = {"theta.ekf": x, "theta.pf": -x*0.5, "measd": y}
    target = {"theta": xd, "meas": yd}

    # Instantiate Evaluator
    metrics = ["mse", "mae", "IAE", "Rmse", "ITAE", "itse", "Var", "std", "ae", "se"]
    evaluator = Evaluator(metrics)

    # Get results
    results = evaluator(data, target, dt=0.2)

    # Plot timeseries
    evaluator.plot(data, target, dt=0.2)
    plt.show()


if __name__ == "__main__":
    import argparse, numpy as np, matplotlib.pyplot as plt
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description="Run the fault diagnosis arena.")
    parser.add_argument("--data", required=True, help="Path to the data.")
    parser.add_argument("--target", required=True, help="Path to the target.")
    args = parser.parse_args()

    data = np.load(Path(args.data).resolve())
    target = np.load(Path(args.target).resolve())
    metrics = ["SE", "AE", "IAE", "ITSE", "ITAE"]
    evaluator = Evaluator(metrics)
    evaluator.plot(data, target, dt=0.2)
    plt.tight_layout()
    plt.show()

    # print(target.keys())
    # plt.plot(target["theta"][2])
    # plt.show()

    

