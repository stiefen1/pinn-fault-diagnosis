from python_vehicle_simulator.vehicles.revolt3 import ReVolt3Dynamics, RevoltThrusterParameters, RevoltParameters3DOF
from src.diagnosis.base import RevoltFaultDiagnosis, register_diagnosis_module

from typing import Tuple, Dict

import numpy as np, torch.nn as nn

from src.diagnosis.learning_based import LearningBasedFaultDiagnosis

class YourGreatPINN(LearningBasedFaultDiagnosis):
    def __init__(
        self,
        states: np.ndarray,
        dt: float,
        cfg: Dict,
        *args,
        dp_mode: bool = False,
        **kwargs
    ):
        super().__init__(states, dt, cfg, *args, dp_mode=dp_mode, **kwargs)

    def __get__(self, states:np.ndarray, *args, **kwargs) -> Tuple[Dict, Dict]:
        """
        Implement forward pass for your model 
        """
        return super().__get__(states, *args, **kwargs)

register_diagnosis_module("YourGreatPINN", YourGreatPINN)