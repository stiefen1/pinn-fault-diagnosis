from python_vehicle_simulator.lib.diagnosis import IDiagnosis
from python_vehicle_simulator.vehicles.revolt3 import RevoltParameters3DOF, RevoltThrusterParameters, ReVolt3Dynamics
from python_vehicle_simulator.lib.weather import Wind, Current
import numpy as np

from typing import Tuple, Dict, Optional, Any

DIAGNOSIS_MODULE_REGISTRY: dict[str, type["RevoltFaultDiagnosis"]] = {}

def register_diagnosis_module(name: str, model_cls: type["RevoltFaultDiagnosis"]) -> None: # You should also add the python file containing your module to src/diagnosis/__init__.py
	DIAGNOSIS_MODULE_REGISTRY[name] = model_cls

def create_diagnosis_module(cfg: dict[str, Any], dt: float) -> "RevoltFaultDiagnosis":
	model_name = str(cfg["name"])
	if model_name not in DIAGNOSIS_MODULE_REGISTRY:
		available = ", ".join(sorted(DIAGNOSIS_MODULE_REGISTRY.keys()))
		raise ValueError(f"Unknown model '{model_name}'. Available: {available}")

	model_cls = DIAGNOSIS_MODULE_REGISTRY[model_name]
	diagnosis_module = model_cls(dt=dt, **cfg["kwargs"])
	return diagnosis_module

class RevoltFaultDiagnosis(IDiagnosis):
    actuator_params: RevoltThrusterParameters = RevoltThrusterParameters()
    dynamics: ReVolt3Dynamics

    def __init__(
            self,
            states: np.ndarray,
            dt:float,
            *args,
            dp_mode: bool = False, 
            **kwargs
    ):
        self.dynamics = ReVolt3Dynamics(dt, dp_mode=dp_mode)
        super().__init__(states, RevoltParameters3DOF(), dt, *args, **kwargs)

    def __get__(self, states:np.ndarray, *args, **kwargs) -> Tuple[Dict, Dict]:
        raise NotImplementedError(f"you must implement a __get__ method returning fault diagnosis")

    # --- helpers ---
    def compute_disturbance(self, ext_state: np.ndarray, wind: Wind, current: Current) -> np.ndarray:
        """Wind + current disturbance force from a single 18-dim ext state."""
        if self.dynamics.vessel_params is None:
            return np.zeros(3)
        p = self.dynamics.vessel_params
        psi, u_vel, v_vel, r = ext_state[5], ext_state[6], ext_state[7], ext_state[11]
        uw, vw = wind.u(psi), wind.v(psi)
        u_rw, v_rw = uw - u_vel, vw - v_vel
        gamma_w = wind.gamma_w(psi)
        wind_rw2 = u_rw**2 + v_rw**2
        tau_coeff = 0.5 * wind.get_air_density() * wind_rw2
        tau_w = np.array([
            tau_coeff * (-p.cx * np.cos(gamma_w)) * p.proj_area_f,
            tau_coeff * ( p.cy * np.sin(gamma_w)) * p.proj_area_l,
            tau_coeff * ( p.cn * np.sin(2 * gamma_w)) * p.proj_area_l * p.loa,
        ])
        uvr = np.array([u_vel, v_vel, r])
        v_c = np.array([current.u(psi), current.v(psi), 0.0])
        tau_c = p.CA(uvr) @ uvr - p.CA(uvr - v_c) @ (uvr - v_c) + p.D @ v_c
        return tau_c + tau_w

    def measurement_model(self, states:np.ndarray) -> np.ndarray:
        return np.take(states, (0, 1, 5, 6, 7, 11, 12, 13)).squeeze()

    def residuals(self, x_hat:np.ndarray, y:np.ndarray) -> np.ndarray:
        y_hat = self.measurement_model(x_hat)
        return y_hat - y[0:8].squeeze()

    def fault_indicator(self, x_hat:np.ndarray, y:np.ndarray, S:np.ndarray) -> float:
        r = self.residuals(x_hat, y)
        return float((r.T @ np.linalg.pinv(S[0:8, 0:8]) @ r).astype(float))

    def prediction_error(self, states: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.measurement_model(states) - y[0:8].squeeze()
    
    def predict(self, states:np.ndarray, u:np.ndarray, wind:Wind, current:Current, theta:Optional[np.ndarray]=None) -> np.ndarray:
        """wrapper to convert wind and current into disturbances and call dynamics.fd"""
        if self.dynamics.vessel_params is not None:    
            disturbance = self.compute_disturbance(states, wind, current)
        else:
            disturbance = np.array(3*[0.0])
            
        if theta is None:
            theta = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        if len(theta.shape) > 1:
            N = theta.shape[0]
            return self.dynamics.fd_batch(np.repeat([states], N, axis=0), np.repeat([u], N, axis=0), theta=theta, disturbance=np.repeat([disturbance], N, axis=0))
        return self.dynamics.fd(states, u, theta=theta, disturbance=disturbance)
    
