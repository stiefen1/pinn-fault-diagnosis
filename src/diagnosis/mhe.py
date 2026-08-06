from python_vehicle_simulator.vehicles.revolt3 import ReVolt3Dynamics, RevoltThrusterParameters, RevoltParameters3DOF
from src.diagnosis.base import RevoltFaultDiagnosis

from typing import Tuple, Dict

import numpy as np, casadi as cs

class MHEFaultDiagnosis(RevoltFaultDiagnosis):
    def __init__(
            self,
            states: np.ndarray,
            dt: float,
            horizon: int,
            *args,
            dp_mode: bool = False,
            **kwargs
    ):
        self.horizon = horizon
        super().__init__(states, dt, *args, dp_mode=dp_mode, **kwargs)

    def __get__(self, states:np.ndarray, *args, **kwargs) -> Tuple[Dict, Dict]:
        return {}, {}