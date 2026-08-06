from python_vehicle_simulator.vehicles.revolt3 import ReVolt3Dynamics, RevoltThrusterParameters, RevoltParameters3DOF
from src.diagnosis.base import RevoltFaultDiagnosis

from typing import Tuple, Dict

import numpy as np, torch.nn as nn

class NNFaultDiagnosis(RevoltFaultDiagnosis):
    model: nn.Module

    def __init__(
            self,
            states: np.ndarray,
            dt: float,
            path_to_weights: str,
            *args,
            dp_mode: bool = False,
            **kwargs
    ):
        super().__init__(states, dt, *args, dp_mode=dp_mode, **kwargs)
        self.load_model(path_to_weights)

    def __get__(self, states:np.ndarray, *args, **kwargs) -> Tuple[Dict, Dict]:
        raise NotImplementedError()
    
    def load_model(self, path_to_weights: str) -> None:
        raise NotImplementedError()
        # self.model = 


class YourGreatPINN(NNFaultDiagnosis):
    def __init__(
        self,
        states: np.ndarray,
        dt: float,
        path_to_weights: str,
        *args,
        dp_mode: bool = False,
        **kwargs
    ):
        super().__init__(states, dt, path_to_weights, *args, dp_mode=dp_mode, **kwargs)

    def __get__(self, states:np.ndarray, *args, **kwargs) -> Tuple[Dict, Dict]:
        """
        Implement forward pass for your model 
        """
        return super().__get__(states, *args, **kwargs)
        # out = self.model(states)
        # return {}, {}

    def load_model(self, path_to_weights: str) -> None:
        # Load your NN module in self.model
        pass