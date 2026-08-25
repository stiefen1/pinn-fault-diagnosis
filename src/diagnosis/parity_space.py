from python_vehicle_simulator.vehicles.revolt3 import ReVolt3Dynamics, RevoltThrusterParameters, RevoltParameters3DOF
from python_vehicle_simulator.lib.weather import Wind, Current
from src.diagnosis.base import RevoltFaultDiagnosis, register_diagnosis_module

from typing import Tuple, Dict, Optional

import numpy as np

class ParitySpaceFaultDiagnosis(RevoltFaultDiagnosis):
    def __init__(
            self,
            dt: float,
            horizon: int,
            *args,
            states: Optional[np.ndarray] = None,
            dp_mode: bool = False,
            **kwargs
    ):
        self.horizon = horizon

        if states is None:
            states = np.array(10*[0.0] + 4*[1.0])
        
        super().__init__(states, dt, *args, dp_mode=dp_mode, **kwargs)

    @staticmethod
    def self_theta_from_self_x(x: np.ndarray) -> np.ndarray:
        return x[10:14]
    
    @staticmethod
    def ext_theta_from_self_x(x: np.ndarray) -> np.ndarray:
        return np.concatenate([x[10:12], [1.0], x[12:14], [1.0]])
    
    @staticmethod
    def ext_u_from_self_u(u: np.ndarray) -> np.ndarray:
        return np.concatenate([u[0:2], [0.0], u[2:4], [0.0]])
    
    @staticmethod
    def self_u_from_ext_u(u: np.ndarray) -> np.ndarray:
        return np.concatenate([u[0:2], u[3:5]])
    
    @staticmethod
    def self_meas_from_ext_meas(z: np.ndarray) -> np.ndarray:
        return z[0:8] # last measurement is alpha3
    
    @staticmethod
    def ext_x_from_self_x(x: np.ndarray) -> np.ndarray:
        """Convert 3DOFs state  into 6DOFs"""
        return np.concatenate([x[0:2], [0, 0, 0, x[2]], x[3:5], [0, 0, 0, x[5]], x[6:8], [0], x[8:10], [0]])
    
    @staticmethod
    def self_x_from_ext_x(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Convert 6DOFs state  into 3DOFs"""
        return np.concatenate([x[0:2], x[5:8], x[11:14], x[15:17], theta])
    
    def h(self, x:np.ndarray) -> np.ndarray:
        return np.concatenate([x[0:]])

    def __get__(self, states:np.ndarray, control_commands:np.ndarray, measurements:np.ndarray, wind: Wind, current: Current, *args, **kwargs) -> Tuple[Dict, Dict]:
        y = self.self_meas_from_ext_meas(measurements)

        # Prediction step
        x_next = self.predict(self.ext_x_from_self_x(self.states), control_commands, wind, current, theta=self.ext_theta_from_self_x(self.states))



        # update self state
        # self.states = ...
        return {}, {}

register_diagnosis_module("ParitySpaceFaultDiagnosis", ParitySpaceFaultDiagnosis)