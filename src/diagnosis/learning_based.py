from python_vehicle_simulator.vehicles.revolt3 import ReVolt3Dynamics, RevoltThrusterParameters, RevoltParameters3DOF
from src.diagnosis.base import RevoltFaultDiagnosis, register_diagnosis_module

from src.architecture.base import create_model, LearningBasedFaultEstimator

from typing import Tuple, Dict, Any

import numpy as np, torch.nn as nn


class LearningBasedFaultDiagnosis(RevoltFaultDiagnosis):
    model: LearningBasedFaultEstimator

    def __init__(
            self,
            states: np.ndarray,
            dt: float,
            cfg: Dict,
            *args,
            dp_mode: bool = False,
            **kwargs
    ):
        super().__init__(states, dt, *args, dp_mode=dp_mode, **kwargs)
        self.load_model(cfg)

    def __get__(self, states:np.ndarray, *args, **kwargs) -> Tuple[Dict, Dict]:
        raise NotImplementedError(f"you must implement a __get__ method returning fault diagnosis")
    
    def load_model(self, cfg: Dict) -> None: # Should be in a base LearningBasedFaultDiagnosis class
        self.model = create_model(cfg)
        self.model.eval()
