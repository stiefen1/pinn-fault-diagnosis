from python_vehicle_simulator.vehicles.revolt3 import ReVolt3Dynamics, RevoltThrusterParameters, RevoltParameters3DOF
from python_vehicle_simulator.lib.weather import Wind, Current
from src.diagnosis.base import register_diagnosis_module
from src.diagnosis.learning_based import LearningBasedFaultDiagnosis

from src.architecture.mlp import MLPFaultEstimator

from typing import Tuple, Dict, Optional

import numpy as np
from torch import Tensor

from copy import deepcopy

class MLPFaultDiagnosis(LearningBasedFaultDiagnosis):
    model: MLPFaultEstimator # type: ignore

    def __init__(
            self,
            dt: float,
            cfg: Dict,
            *args,
            states: Optional[np.ndarray] = None,
            dp_mode: bool = False,
            **kwargs
    ):
        if states is None:
            states = np.array(10*[0.0] + 4*[1.0])

        super().__init__(states, dt, cfg, *args, dp_mode=dp_mode, **kwargs)
        self.input_buffer = []
        self.y_prev = None

    def __get__(self, states: np.ndarray, control_commands: np.ndarray, measurements: np.ndarray, wind: Wind, current: Current, prev_navigation: Dict, *args, **kwargs) -> Tuple[Dict, Dict]:
        prev_wind = prev_navigation["wind_meas"] if "wind_meas" in prev_navigation.keys() else deepcopy(wind)
        prev_current = prev_navigation["current_meas"] if "current_meas" in prev_navigation.keys() else deepcopy(current)               
        diagnosis_theta = np.array(6*[1.0]) # Assume perfect theta if not enough measurements are available 

        if self.y_prev is not None:
            self.input_buffer.append(np.concatenate([self.y_prev.squeeze(), control_commands.squeeze(), measurements.squeeze(), [prev_wind.beta, prev_wind.norm, prev_current.beta, prev_current.norm]]))

        if len(self.input_buffer) >= self.model.n_samples:   
            # print(len(self.input_buffer), )         
            x = Tensor(np.concatenate(self.input_buffer, axis=0)).unsqueeze(0)
            diagnosis_theta = self.model.forward(x).squeeze(0).detach().numpy()
            info = {'active': True}
            self.input_buffer.pop(0)
        else:
            info = {'active': False}

        self.y_prev = measurements.copy()

        return {'diagnosis_theta': diagnosis_theta}, info

register_diagnosis_module("MLPFaultDiagnosis", MLPFaultDiagnosis)
