"""
Ideal usage:

cfg = EvalCfg.from_yaml(path_to_cfg)
env = EvalEnv(cfg)

"""


### TODO:
# - replace _build_env in generator.py by this one
# - Find the best way to run an episode and gather results
# - Instantiate diagnosis module -> find the best way to find the class specified in the configuration file


from src.vessel.guidance import TrajectoryTrackingGuidance
from src.vessel.navigation import NavigationRevolt
from src.vessel.control import NMPCTrajectoryTrackerRevolt
from src.excitation.wrapper import AuxiliaryExcitationWrapper
from src.diagnosis.base import create_diagnosis_module
from src.utils.random import build_fault_schedule

from python_vehicle_simulator.lib.path import PWLPath
from python_vehicle_simulator.vehicles.revolt3 import ReVolt3
from python_vehicle_simulator.lib.weather import Wind, Current
from python_vehicle_simulator.lib.env import NavEnv

import numpy as np

from typing import Dict

def build_env(scenario_cfg: Dict) -> NavEnv:
    """
    Returns the navigation environment described in scenario_cfg
    """
    vessel_cfg = scenario_cfg["vessel"]
    dt = scenario_cfg["dt"]
    seed = scenario_cfg["seed"]

    wind_speed = scenario_cfg["wind"]["speed"]["ornstein_uhlenbeck"]["average"]
    wind_angle = scenario_cfg["wind"]["angle"]["ornstein_uhlenbeck"]["average"]
    current_speed = scenario_cfg["current"]["speed"]["ornstein_uhlenbeck"]["average"]
    current_angle = scenario_cfg["current"]["angle"]["ornstein_uhlenbeck"]["average"]

    horizon = int(vessel_cfg["control"]["horizon"])
    dp_mode = bool(vessel_cfg["dp_mode"])
    aux_exc_cfg = vessel_cfg.get("auxiliary_excitation", {})

    guidance_cfg = vessel_cfg["guidance"]
    desired_speed = guidance_cfg["desired_speed"]
    path = PWLPath.sample(
        d_tot=float(guidance_cfg["d_tot"]),
        max_turn_deg=float(guidance_cfg["max_turn_deg"]),
        seg_len_range=tuple(guidance_cfg["seg_len_range"]),
        seed=seed,
    ).smooth(int(guidance_cfg["smooth_radius"])) # type: ignore

    control_cfg = vessel_cfg["control"]
    control_enabled = control_cfg["enabled"]
    aux_exc_enabled = aux_exc_cfg.get("enabled", False)

    controller = NMPCTrajectoryTrackerRevolt(
                dt=dt,
                dp_mode=dp_mode,
                **control_cfg
            ) if control_enabled else None
    
    if aux_exc_enabled:
        controller = AuxiliaryExcitationWrapper(
            controller=controller,
            cfg=aux_exc_cfg, 
            seed=seed 
        )

    assert controller is not None, f"Either an auxiliary excitation or a controller must be provided."

    diagnosis_cfg = vessel_cfg.get("diagnosis", None)
    diagnosis = create_diagnosis_module(diagnosis_cfg, dt) if diagnosis_cfg is not None else None

    navigation_cfg = vessel_cfg["navigation"]
    if navigation_cfg["enabled"]:
        R_se = np.diag(vessel_cfg["navigation"]["meas_cov"])
        navigation = NavigationRevolt(np.array(18*[0]), dt, dp_mode=dp_mode, seed=seed, R_se=R_se, **navigation_cfg)
    else:
        navigation = None

    vessel = ReVolt3(
        dt,
        dp_mode=dp_mode,
        control=controller,
        guidance=TrajectoryTrackingGuidance( # guidance is always there even though it might be unused
            path,
            desired_speed,
            dt,
            horizon,
        ),
        navigation=navigation,
        diagnosis=diagnosis
    )

    env = NavEnv(
        own_vessel=vessel,
        target_vessels=[],
        obstacles=[],
        dt=dt,
        current=Current(
            beta=current_angle,
            v=current_speed,
            attraction_beta=float(scenario_cfg["current"]["angle"]["ornstein_uhlenbeck"]["attraction"]),
            amplitude_beta=float(scenario_cfg["current"]["angle"]["ornstein_uhlenbeck"]["amplitude"]),
            attraction_norm=float(scenario_cfg["current"]["speed"]["ornstein_uhlenbeck"]["attraction"]),
            amplitude_norm=float(scenario_cfg["current"]["speed"]["ornstein_uhlenbeck"]["amplitude"]),
            dt=dt,
            seed=seed
        ),
        wind=Wind(
            beta=wind_angle,
            v=wind_speed,
            attraction_beta=float(scenario_cfg["wind"]["angle"]["ornstein_uhlenbeck"]["attraction"]),
            amplitude_beta=float(scenario_cfg["wind"]["angle"]["ornstein_uhlenbeck"]["amplitude"]),
            attraction_norm=float(scenario_cfg["wind"]["speed"]["ornstein_uhlenbeck"]["attraction"]),
            amplitude_norm=float(scenario_cfg["wind"]["speed"]["ornstein_uhlenbeck"]["amplitude"]),
            dt=dt,
            seed=seed
        ),
    )
    env.reset(seed=seed)
    return env

def run_episode(scenario_cfg: dict, max_steps: int, nx: int, nu: int, ny: int, ntheta: int) -> dict[str, np.ndarray]:
    """
    Run a single episode based on scenario_cfg with max_steps.
    """
    env = build_env(scenario_cfg)
    vessel = env.own_vessel

    faults_cfg = scenario_cfg["vessel"]["faults"]
    theta_schedule = build_fault_schedule(faults_cfg, max_steps, healthy_value=1.0)
    diagnosis_enabled = scenario_cfg["vessel"].get("diagnosis", {}).get("enabled", False)
    navigation_enabled = scenario_cfg["vessel"].get("navigation", {}).get("enabled", False)

    x = np.zeros((max_steps + 1, nx), dtype=np.float32) # actual state (inaccessible in real-world)
    x_hat = np.zeros((max_steps, nx), dtype=np.float32) # state estimation (from e.g. EKF)
    y = np.zeros((max_steps, ny), dtype=np.float32) # sensor measurements
    u = np.zeros((max_steps, nu), dtype=np.float32)
    wind = np.zeros((max_steps, 2), dtype=np.float32) # max_steps * [beta, norm]
    current = np.zeros((max_steps, 2), dtype=np.float32)
    y_wind = np.zeros((max_steps, 2), dtype=np.float32)
    y_current = np.zeros((max_steps, 2), dtype=np.float32)
    ne_des = np.zeros((max_steps, 2), dtype=np.float32)

    # Faults
    theta = np.zeros((max_steps, ntheta), dtype=np.float32)
    diagnosis_theta = np.zeros((max_steps, ntheta), dtype=np.float32)

    x[0] = vessel.states.copy()

    t_sec = 0.0
    verbose = scenario_cfg.get("verbose", 10)
    for k in range(max_steps):
        theta_k = theta_schedule[k]
        env.step(theta=theta_k, t=t_sec)
        if scenario_cfg.get("render", False):
            env.render(mode="human", verbose=verbose)

        x[k + 1] = vessel.states.copy()
        u[k] = np.asarray(vessel.control.prev["u"], dtype=np.float32)
        theta[k] = theta_k.astype(np.float32)
        wind_k: Wind = vessel.navigation.prev["wind"]
        wind[k] = np.array([wind_k.beta, wind_k.norm], dtype=np.float32)
        current_k: Current = vessel.navigation.prev["current"]
        current[k] = np.array([current_k.beta, current_k.norm], dtype=np.float32)
        ne_des[k] = np.array([vessel.guidance.prev["info"]["ne_des"]], dtype=np.float32)

        if diagnosis_enabled:
            diagnosis_k =  vessel.diagnosis.prev["diagnosis"]
            assert diagnosis_k is not None, f"previous diagnosis is None"
            diagnosis_theta[k] = np.asarray(diagnosis_k["diagnosis_theta"])

        if navigation_enabled:
            y[k] = np.asarray(vessel.navigation.prev["measurements"], dtype=np.float32)
            x_hat[k] = np.asarray(vessel.navigation.prev["states_est"])
            y_wind_k: Wind = vessel.navigation.prev["wind_meas"]
            y_wind[k] = np.array([y_wind_k.beta, y_wind_k.norm], dtype=np.float32)
            y_current_k: Current = vessel.navigation.prev["current_meas"]
            y_current[k] = np.array([y_current_k.beta, y_current_k.norm], dtype=np.float32)


        guidance_info = vessel.guidance.prev.get("info", {})
        if bool(guidance_info.get("term", False)):
            x = x[: k + 2]
            y = y[: k + 1]
            u = u[: k + 1]
            theta = theta[: k + 1]
            x_hat = x[ : k + 1]
            wind = wind[ : k + 1]
            current = current[ : k + 1]
            y_wind = y_wind[ : k + 1]
            y_current = y_current[ : k + 1]
            ne_des = ne_des[ : k + 1]
            break

        t_sec += env.dt

    out = {
        "x": x,
        "u": u,
        "theta": theta,
        "wind": wind,
        "current": current,
        "ne_des": ne_des
    }

    if navigation_enabled:
        out = out | {"y": y, "x_hat": x_hat, "y_wind": y_wind, "y_current": y_current}

    if diagnosis_enabled:
        out = out | {"diagnosis_theta": diagnosis_theta}

    return out

if __name__ == "__main__":
    from src.env.config import EnvCfg
    import pathlib
    path_to_cfg = pathlib.Path("configs/eval.yaml")
    cfg = EnvCfg.from_yaml(path_to_cfg=path_to_cfg)
    scenario_cfg = cfg(n=1)[0]
    env = build_env(scenario_cfg)