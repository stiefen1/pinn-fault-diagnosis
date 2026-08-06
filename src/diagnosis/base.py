from python_vehicle_simulator.lib.diagnosis import IDiagnosis
from python_vehicle_simulator.vehicles.revolt3 import RevoltParameters3DOF, RevoltThrusterParameters, ReVolt3Dynamics
from python_vehicle_simulator.lib.weather import Wind, Current
import numpy as np

from typing import Tuple, Dict, Optional

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
    
    def predict(self, states:np.ndarray, u:np.ndarray, wind:Wind, current:Current, theta:Optional[np.ndarray]=None) -> np.ndarray:
        """wrapper to convert wind and current into disturbances and call dynamics.fd"""
        if self.dynamics.vessel_params is not None:    
            # Wind perturbations
            uw = wind.u(states[5])
            vw = wind.v(states[5])

            u_rw = uw - states[6]
            v_rw = vw - states[7]

            gamma_w = wind.gamma_w(states[5])
            wind_rw2 = u_rw**2 + v_rw**2
            c_x = -self.dynamics.vessel_params.cx * np.cos(gamma_w)
            c_y = self.dynamics.vessel_params.cy * np.sin(gamma_w)
            c_n = self.dynamics.vessel_params.cn * np.sin(2 * gamma_w)

            tau_coeff = 0.5 * wind.get_air_density() * wind_rw2
            tau_w = np.array([
                tau_coeff * c_x * self.dynamics.vessel_params.proj_area_f,
                tau_coeff * c_y * self.dynamics.vessel_params.proj_area_l,
                tau_coeff * c_n * self.dynamics.vessel_params.proj_area_l * self.dynamics.vessel_params.loa
            ]) 

            # Current perturbations
            uvr = np.take(states, [6, 7, 11])
            v_c = np.array([current.u(states[5]), current.v(states[5]), 0]) # current speed in ship frame
            tau_c_coriolis = self.dynamics.vessel_params.CA(uvr) @ uvr - self.dynamics.vessel_params.CA(uvr - v_c) @ (uvr - v_c) # cancel CA(nu) @ nu and add CA(nu_r) @ nu_r
            tau_c_damping = self.dynamics.vessel_params.D @ v_c
            tau_c = tau_c_coriolis + tau_c_damping
            disturbance = tau_c + tau_w
        else:
            disturbance = np.array(3*[0.0])
            
        if theta is None:
            theta = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        if len(theta.shape) > 1:
            N = theta.shape[0]
            return self.dynamics.fd_batch(np.repeat([states], N, axis=0), np.repeat([u], N, axis=0), theta=theta, disturbance=np.repeat([disturbance], N, axis=0))
        return self.dynamics.fd(states, u, theta=theta, disturbance=disturbance)
    
    
    