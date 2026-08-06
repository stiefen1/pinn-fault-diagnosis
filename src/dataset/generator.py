from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from python_vehicle_simulator.lib.env import NavEnv
from python_vehicle_simulator.lib.path import PWLPath
from python_vehicle_simulator.lib.weather import Current, Wind
from python_vehicle_simulator.vehicles.revolt3 import ReVolt3

from src.utils.configs import load_config
from src.utils.random import sample_clipped_value, sample_uniform_min_max
from src.vessel.control import NMPCTrajectoryTrackerRevolt
from src.vessel.guidance import TrajectoryTrackingGuidance
from src.vessel.navigation import NavigationRevolt

# from src.excitation.base import ExcitationSignal1D
from src.excitation.wrapper import AuxiliaryExcitationWrapper


class FaultIdentificationDatasetGenerator:
    def __init__(self, config_path: str | Path, render: bool = False):
        self.config_path = Path(config_path).resolve()
        self.cfg = load_config(self.config_path)
        features_cfg = self.cfg["dataset"]["features"]
        self.nx = int(features_cfg["nx"])
        self.nu = int(features_cfg["nu"])
        self.ntheta = int(features_cfg["ntheta"])
        self.render = render

    @staticmethod
    def _build_fault_schedule(
        rng: np.random.Generator,
        cfg_fault: dict[str, Any],
        max_steps: int,
        healthy_value: float,
    ) -> np.ndarray:
        schedule = np.full((max_steps, 3), healthy_value, dtype=np.float32) # n_steps x ntheta -> fault parameter along time vector

        probs = np.asarray(cfg_fault["prob"], dtype=float) # tuple[p1, p2, p3]

        amp_cfg = cfg_fault["amplitude"] # fault amplitude \in [0, 1]
        time_cfg = cfg_fault["time"] # parameters of uniform or clipped gaussian distribution
        is_integer = cfg_fault.get("is_integer", False)

        for i in range(3):
            # Independent Bernoulli trial per actuator for this fault type.
            if rng.random() < probs[i]:
                t_norm = sample_clipped_value(rng, time_cfg) # randomy sample fault time, normalized \in [0, 1] i.e equals 1 if t_fault = tf
                k_fault = int(np.floor(t_norm * (max_steps - 1))) # map normalized time to time-step
                amp = sample_clipped_value(rng, amp_cfg) # randomly sample fault amplitude
                schedule[k_fault:, i] = int(amp) if is_integer else amp # theta only changes once along a trajectory
        return schedule

    def _sample_theta_schedule(self, rng: np.random.Generator, max_steps: int) -> np.ndarray:
        faults_cfg = self.cfg["vessel"]["faults"]

        azimuth_sched = self._build_fault_schedule(
            rng,
            faults_cfg["azimuth_stucked"],
            max_steps=max_steps,
            healthy_value=1.0,
        )
        loe_sched = self._build_fault_schedule(
            rng,
            faults_cfg["loss_of_effectiveness"],
            max_steps=max_steps,
            healthy_value=1.0,
        )

        theta_schedule = np.concatenate([azimuth_sched, loe_sched], axis=1)
        return theta_schedule.astype(np.float32)

    def _build_env(self, episode_seed: int) -> NavEnv:
        rng = np.random.default_rng(episode_seed)

        vessel_cfg = self.cfg["vessel"]
        env_cfg = self.cfg["env"]
        self.render = bool(env_cfg.get("render", self.render))

        wind_cfg = env_cfg["wind"]
        current_cfg = env_cfg["current"]

        wind_speed_cfg = wind_cfg["speed"]
        wind_angle_cfg = wind_cfg["angle"]
        current_speed_cfg = current_cfg["speed"]
        current_angle_cfg = current_cfg["angle"]

        wind_speed = sample_uniform_min_max(rng, wind_speed_cfg)
        wind_angle = sample_uniform_min_max(rng, wind_angle_cfg)
        current_speed = sample_uniform_min_max(rng, current_speed_cfg)
        current_angle = sample_uniform_min_max(rng, current_angle_cfg)

        dt = float(vessel_cfg["dt"])
        horizon = int(vessel_cfg["control"]["horizon"])
        dp_mode = bool(vessel_cfg["dp_mode"])
        aux_exc_cfg = vessel_cfg["auxiliary_excitation"]

        guidance_cfg = vessel_cfg["guidance"]
        target_speed = sample_clipped_value(rng, guidance_cfg["target_speed"])
        path = PWLPath.sample(
            d_tot=float(guidance_cfg["d_tot"]),
            max_turn_deg=float(guidance_cfg["max_turn_deg"]),
            seg_len_range=tuple(guidance_cfg["seg_len_range"]),
            seed=episode_seed,
        ).smooth(int(guidance_cfg["smooth"]))

        control_cfg = vessel_cfg["control"]
        control_enabled = control_cfg["enabled"]
        aux_exc_enabled = aux_exc_cfg["enabled"]

        controller = NMPCTrajectoryTrackerRevolt(
                    dt=dt,
                    dp_mode=dp_mode,
                    **control_cfg
                ) if control_enabled else None
        
        if aux_exc_enabled:
            controller = AuxiliaryExcitationWrapper(
                controller=controller,
                cfg=aux_exc_cfg, 
                seed=episode_seed 
            )

        navigation_cfg = vessel_cfg["navigation"]
        navigation = NavigationRevolt(np.array(18*[0]), dt, dp_mode=dp_mode, seed=episode_seed, **navigation_cfg) if navigation_cfg["enabled"] else None

        assert controller is not None, f"Either an auxiliary excitation or a controller must be provided."

        vessel = ReVolt3(
            dt,
            dp_mode=dp_mode,
            control=controller,
            guidance=TrajectoryTrackingGuidance( # guidance is always there even though it might be unused
                path,
                target_speed,
                dt,
                horizon,
            ),
            navigation=navigation
        )

        env = NavEnv(
            own_vessel=vessel,
            target_vessels=[],
            obstacles=[],
            dt=dt,
            current=Current(
                beta=current_angle,
                v=current_speed,
                attraction_beta=float(current_angle_cfg["ornstein_uhlenbeck"]["attraction"]),
                amplitude_beta=float(current_angle_cfg["ornstein_uhlenbeck"]["amplitude"]),
                attraction_norm=float(current_speed_cfg["ornstein_uhlenbeck"]["attraction"]),
                amplitude_norm=float(current_speed_cfg["ornstein_uhlenbeck"]["amplitude"]),
                dt=dt,
                seed=episode_seed
            ),
            wind=Wind(
                beta=wind_angle,
                v=wind_speed,
                attraction_beta=float(wind_angle_cfg["ornstein_uhlenbeck"]["attraction"]),
                amplitude_beta=float(wind_angle_cfg["ornstein_uhlenbeck"]["amplitude"]),
                attraction_norm=float(wind_speed_cfg["ornstein_uhlenbeck"]["attraction"]),
                amplitude_norm=float(wind_speed_cfg["ornstein_uhlenbeck"]["amplitude"]),
                dt=dt,
                seed=episode_seed
            ),
        )
        env.reset(seed=episode_seed)
        return env

    def _run_episode(self, episode_seed: int) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(episode_seed)
        env = self._build_env(episode_seed)
        vessel = env.own_vessel

        max_steps = int(self.cfg["episodes"]["max_steps_per_episode"])
        theta_schedule = self._sample_theta_schedule(rng, max_steps)

        x = np.zeros((max_steps + 1, self.nx), dtype=np.float32)
        y = np.zeros((max_steps, self.nx), dtype=np.float32)
        u = np.zeros((max_steps, self.nu), dtype=np.float32)
        theta = np.zeros((max_steps, self.ntheta), dtype=np.float32)

        x[0] = vessel.states.copy()

        t_sec = 0.0
        for k in range(max_steps):
            theta_k = theta_schedule[k]
            env.step(theta=theta_k, t=t_sec)
            if self.render:
                env.render(mode="human", verbose=10)

            x[k + 1] = vessel.states.copy()
            y[k] = np.asarray(vessel.navigation.prev["states"], dtype=np.float32)
            u[k] = np.asarray(vessel.control.prev["u"], dtype=np.float32)
            theta[k] = theta_k.astype(np.float32)

            guidance_info = vessel.guidance.prev.get("info", {})
            if bool(guidance_info.get("term", False)):
                x = x[: k + 2]
                y = y[: k + 1]
                u = u[: k + 1]
                theta = theta[: k + 1]
                break

            t_sec += env.dt

        return {"x": x, "y": y, "u": u, "theta": theta}

    def generate(self) -> Path:
        seed_cfg = self.cfg["seed"]
        base_seed = int(seed_cfg["global"])
        n_episodes = int(self.cfg["episodes"]["N"])
        num_workers = int(self.cfg["episodes"].get("num_workers", 1))

        seeds = [base_seed + ep for ep in range(n_episodes)]

        if num_workers > 1:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                episodes = list(tqdm(executor.map(self._run_episode, seeds), total=n_episodes, desc="Episodes"))
        else:
            episodes = [self._run_episode(s) for s in tqdm(seeds, desc="Episodes")]

        lengths = [len(ep["y"]) for ep in episodes]

        max_len = max(lengths)

        X = np.zeros((n_episodes, max_len + 1, self.nx), dtype=np.float32)
        Y = np.zeros((n_episodes, max_len, self.nx), dtype=np.float32)
        U = np.zeros((n_episodes, max_len, self.nu), dtype=np.float32)
        THETA = np.ones((n_episodes, max_len, self.ntheta), dtype=np.float32)

        for ep, episode in enumerate(episodes):
            t = lengths[ep]
            X[ep, : t + 1] = episode["x"]
            Y[ep, :t] = episode["y"]
            U[ep, :t] = episode["u"]
            THETA[ep, :t] = episode["theta"]

        dataset_cfg = self.cfg["dataset"]
        save_dir = Path(dataset_cfg["save_dir"])
        file_name = str(dataset_cfg["file_name"])
        save_path = (save_dir / file_name).resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            save_path,
            x=X,
            y=Y,
            u=U,
            theta=THETA,
            lengths=np.asarray(lengths, dtype=np.int32),
        )
        return save_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fault-identification trajectory dataset")
    parser.add_argument("--config", "-c", type=str, default="configs/dataset.yaml")
    args = parser.parse_args()

    generator = FaultIdentificationDatasetGenerator(args.config, True)
    out_path = generator.generate()
    print(f"Saved dataset to: {out_path}")


if __name__ == "__main__":
    main()