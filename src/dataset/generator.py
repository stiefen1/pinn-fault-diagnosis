from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

from src.utils.configs import load_config
from src.env.core import run_episode
from src.env.config import EnvCfg

# from src.excitation.base import ExcitationSignal1D
from src.excitation.wrapper import AuxiliaryExcitationWrapper


class FaultIdentificationDatasetGenerator:
    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path).resolve()
        self.cfg = load_config(self.config_path)
        features_cfg = self.cfg["dataset"]["features"]
        self.ny = int(features_cfg["ny"])
        self.nu = int(features_cfg["nu"])
        self.nx = int(features_cfg["nx"])
        self.ntheta = int(features_cfg["ntheta"])

    def generate(self, cfg_save_dir: Optional[str] = None, cfg_filenames: str = 'scenario_') -> Path:
        n_episodes = int(self.cfg["episodes"]["N"])
        num_workers = int(self.cfg["episodes"].get("num_workers", 1))

        env_cfg = EnvCfg.from_yaml(self.config_path)
        scenarios = env_cfg(n_episodes, save_dir=cfg_save_dir, filenames=cfg_filenames)
        max_steps = int(self.cfg["episodes"]["max_steps_per_episode"])
        n_scenarios = len(scenarios)

        if num_workers > 1:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                episodes = list(tqdm(executor.map(
                    run_episode,
                    scenarios,
                    n_scenarios*[max_steps],
                    n_scenarios*[self.nx],
                    n_scenarios*[self.nu],
                    n_scenarios*[self.ny],
                    n_scenarios*[self.ntheta]), total=n_episodes, desc="Episodes"))
        else:
            episodes = [run_episode(s, max_steps, self.nx, self.ny, self.nu, self.ntheta) for s in tqdm(scenarios, desc="Episodes")]

        lengths = [len(ep["u"]) for ep in episodes]

        max_len = max(lengths)

        # DEFAULT
        X = np.zeros((n_episodes, max_len + 1, self.nx), dtype=np.float32)
        U = np.zeros((n_episodes, max_len, self.nu), dtype=np.float32)
        THETA = np.ones((n_episodes, max_len, self.ntheta), dtype=np.float32)
        CURRENT = np.ones((n_episodes, max_len, 2), dtype=np.float32)
        WIND = np.ones((n_episodes, max_len, 2), dtype=np.float32)

        # NAVIGATION
        X_HAT = np.zeros((n_episodes, max_len, self.nx), dtype=np.float32)
        Y = np.zeros((n_episodes, max_len, self.ny), dtype=np.float32)
        CURRENT_MEAS = np.ones((n_episodes, max_len, 2), dtype=np.float32)
        WIND_MEAS = np.ones((n_episodes, max_len, 2), dtype=np.float32)

        # DIAGNOSIS
        DIAGNOSIS_THETA = np.zeros((n_episodes, max_len, self.ntheta), dtype=np.float32)

        # GUIDANCE
        NE_DES = np.zeros((n_episodes, max_len, 2), dtype=np.float32)
        
        for ep, episode in enumerate(episodes):
            navigation_enabled = scenarios[ep]["vessel"].get("navigation", {}).get("enabled", False)
            diagnosis_enabled = scenarios[ep]["vessel"].get("diagnosis", {}).get("enabled", False)
            t = lengths[ep]
            X[ep, : t + 1] = episode["x"]
            U[ep, :t] = episode["u"]
            THETA[ep, :t] = episode["theta"]
            CURRENT[ep, :t] = episode["current"]
            WIND[ep, :t] = episode["wind"]
            NE_DES[ep, :t] = episode["ne_des"]

            if navigation_enabled:
                Y[ep, :t] = episode["y"]
                X_HAT[ep, :t] = episode["x_hat"]
                CURRENT_MEAS[ep, :t] = episode["y_current"]
                WIND_MEAS[ep, :t] = episode["y_wind"]

            if diagnosis_enabled:
                DIAGNOSIS_THETA[ep, :t] = episode["diagnosis_theta"]

        dataset_cfg = self.cfg["dataset"]
        save_dir = Path(dataset_cfg["save_dir"])
        file_name = str(dataset_cfg["file_name"])
        save_path = (save_dir / file_name).resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "x": X,
            "u": U,
            "theta": THETA,
            "wind": WIND,
            "current": CURRENT,
            "ne_des": NE_DES
        }

        if navigation_enabled: # type: ignore
            data = data | {"y": Y, "x_hat": X_HAT, "current_meas": CURRENT_MEAS, "wind_meas": WIND_MEAS}

        if diagnosis_enabled: # type: ignore
            data = data | {"diagnosis_theta": DIAGNOSIS_THETA}

        np.savez_compressed(
            save_path,
            **data, # type: ignore
            lengths=np.asarray(lengths, dtype=np.int32),
        )
        return save_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fault-identification trajectory dataset")
    parser.add_argument("--config", "-c", type=str, default="configs/dataset.yaml")
    args = parser.parse_args()

    generator = FaultIdentificationDatasetGenerator(args.config)
    out_path = generator.generate()
    print(f"Saved dataset to: {out_path}")


if __name__ == "__main__":
    main()