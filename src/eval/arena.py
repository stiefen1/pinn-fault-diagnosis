import copy
from typing import Dict, Optional, List
from pathlib import Path

from src.utils.configs import load_config
from src.env.config import EnvCfg
from src.env.core import run_episode
from src.eval.metrics import Evaluator

from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

import numpy as np


def _inject_diagnosis(scenario: dict, name: str, kwargs: dict) -> dict:
    """Return a deep-copied scenario with the diagnosis algorithm overridden."""
    s = copy.deepcopy(scenario)
    s["vessel"]["diagnosis"]["name"] = name
    s["vessel"]["diagnosis"]["kwargs"] = kwargs
    return s


class FaultDiagnosisArena:
    def __init__(
            self,
            config_path: str | Path
        ):
        self.config_path = Path(config_path).resolve()
        self.cfg = load_config(self.config_path)
        features_cfg = self.cfg["dataset"]["features"]
        self.nx = int(features_cfg["nx"])
        self.nu = int(features_cfg["nu"])
        self.ny = int(features_cfg["ny"])
        self.ntheta = int(features_cfg["ntheta"])

    def _run_episodes(self, scenarios: list, max_steps: int, num_workers: int) -> list:
        n = len(scenarios)
        if num_workers > 1:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                return list(tqdm(executor.map(
                    run_episode, scenarios,
                    n * [max_steps], n * [self.nx], n * [self.nu], n * [self.ny], n * [self.ntheta],
                ), total=n, desc="Episodes"))
        return [run_episode(s, max_steps, self.nx, self.nu, self.ny, self.ntheta)
                for s in tqdm(scenarios, desc="Episodes")]

    def _pack_sim_data(self, episodes: list, scenarios: list, max_len: int) -> dict:
        n = len(episodes)
        lengths = [len(ep["u"]) for ep in episodes]
        nav_enabled = scenarios[0]["vessel"].get("navigation", {}).get("enabled", False)

        X = np.zeros((n, max_len + 1, self.nx), dtype=np.float32)
        U = np.zeros((n, max_len, self.nu), dtype=np.float32)
        THETA = np.ones((n, max_len, self.ntheta), dtype=np.float32)
        CURRENT = np.ones((n, max_len, 2), dtype=np.float32)
        WIND = np.ones((n, max_len, 2), dtype=np.float32)
        NE_DES = np.zeros((n, max_len, 2), dtype=np.float32)
        X_HAT = np.zeros((n, max_len, self.nx), dtype=np.float32)
        Y = np.zeros((n, max_len, self.ny), dtype=np.float32)
        CURRENT_MEAS = np.ones((n, max_len, 2), dtype=np.float32)
        WIND_MEAS = np.ones((n, max_len, 2), dtype=np.float32)

        for ep, episode in enumerate(episodes):
            t = lengths[ep]
            X[ep, :t + 1] = episode["x"]
            U[ep, :t] = episode["u"]
            THETA[ep, :t] = episode["theta"]
            CURRENT[ep, :t] = episode["current"]
            WIND[ep, :t] = episode["wind"]
            NE_DES[ep, :t] = episode["ne_des"]
            if nav_enabled:
                Y[ep, :t] = episode["y"]
                X_HAT[ep, :t] = episode["x_hat"]
                CURRENT_MEAS[ep, :t] = episode["y_current"]
                WIND_MEAS[ep, :t] = episode["y_wind"]

        data = {"x": X, "u": U, "theta": THETA, "wind": WIND, "current": CURRENT, "ne_des": NE_DES,
                "lengths": np.asarray(lengths, dtype=np.int32)}
        if nav_enabled:
            data |= {"y": Y, "x_hat": X_HAT, "current_meas": CURRENT_MEAS, "wind_meas": WIND_MEAS}
        return data

    def _pack_diagnosis_theta(self, episodes: list, max_len: int) -> np.ndarray:
        n = len(episodes)
        D = np.zeros((n, max_len, self.ntheta), dtype=np.float32)
        for ep, episode in enumerate(episodes):
            t = len(episode["u"])
            D[ep, :t] = episode["diagnosis_theta"][:t]
        return D

    def generate(self, cfg_save_dir: Optional[str] = None, cfg_filenames: str = 'scenario_') -> Path:
        n_episodes = int(self.cfg["episodes"]["N"])
        num_workers = int(self.cfg["episodes"].get("num_workers", 1))
        max_steps = int(self.cfg["episodes"]["max_steps_per_episode"])

        env_cfg = EnvCfg.from_yaml(self.config_path)
        scenarios = env_cfg(n_episodes, save_dir=cfg_save_dir, filenames=cfg_filenames)

        algo_cfgs: dict = self.cfg.get("diagnosis", {}).get("algorithms", {})

        save_dir = Path(self.cfg["dataset"]["save_dir"])
        data_dir = save_dir / "data"
        metrics_dir = save_dir / "metrics"
        data_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)

        sim_data: Optional[dict] = None
        all_diagnosis: dict[str, np.ndarray] = {}

        for label, algo_cfg in algo_cfgs.items():
            print(f"\n--- Running algorithm: {label} ({algo_cfg['name']}) ---")
            injected = [_inject_diagnosis(s, algo_cfg["name"], algo_cfg.get("kwargs", {}))
                        for s in scenarios]
            episodes = self._run_episodes(injected, max_steps, num_workers)
            lengths = [len(ep["u"]) for ep in episodes]
            max_len = max(lengths)

            if sim_data is None:
                sim_data = self._pack_sim_data(episodes, injected, max_len)

            # Use the canonical max_len from the first run for consistency
            canonical_max_len = sim_data["theta"].shape[1]
            all_diagnosis[label] = self._pack_diagnosis_theta(episodes, canonical_max_len)

        if sim_data is None:
            # No algorithms configured: single run without diagnosis
            episodes = self._run_episodes(scenarios, max_steps, num_workers)
            lengths = [len(ep["u"]) for ep in episodes]
            sim_data = self._pack_sim_data(episodes, scenarios, max(lengths))

        np.savez_compressed(data_dir / "simulation.npz", **sim_data)

        if all_diagnosis:
            np.savez_compressed(data_dir / "diagnosis.npz",
                                **{f"theta.{label}": arr for label, arr in all_diagnosis.items()})

            metric_keys: list = self.cfg.get("episodes", {}).get("metrics", {}).get("diagnosis", [])
            if metric_keys:
                evaluator = Evaluator(metric_keys)
                data_for_metrics = {f"theta.{label}": arr for label, arr in all_diagnosis.items()}
                target_for_metrics = {"theta": sim_data["theta"]}
                results = evaluator(data_for_metrics, target_for_metrics)
                for metric_name, metric_res in results.items():
                    np.savez_compressed(metrics_dir / f"{metric_name}.npz", **metric_res)

        return save_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the fault diagnosis arena.")
    parser.add_argument("-c", "--config", required=True, help="Path to the eval config YAML.")
    args = parser.parse_args()

    arena = FaultDiagnosisArena(args.config)
    out = arena.generate()
    print(f"\nResults saved to: {out}")