from python_vehicle_simulator.lib.control import IControl
from python_vehicle_simulator.lib.weather import Current, Wind
from python_vehicle_simulator.lib.obstacle import Obstacle

from src.excitation.base import ExcitationSignal1D
from src.excitation.signals import Zero

from typing import List, Optional, Dict, Tuple

import numpy as np, numpy.typing as npt

class AuxiliaryExcitationWrapper(IControl):
    def __init__(
            self,
            controller: Optional[IControl] = None,
            cfg: Optional[Dict] = None,
            initial_commands: Optional[npt.NDArray] = None,
            seed: Optional[int] = None,
            **kwargs
    ):
        self.controller = controller
        cfg = cfg if cfg is not None else {}

        candidate_keys = ['port', 'starboard', 'bow'] # order is important here, it matches Revolt parameters in python_vehicle_simulator/vehicles/revolt3.py
        auxiliary_azimuths = []
        auxiliary_speeds = []

        for candidate_key in candidate_keys:
            thruster_cfg = cfg.get(candidate_key, {})

            azimuth_expr = thruster_cfg.get('azimuth', '')
            if isinstance(azimuth_expr, str) and azimuth_expr.strip():
                auxiliary_azimuths.append(ExcitationSignal1D.from_str(azimuth_expr, {'seed': seed}))
            else:
                auxiliary_azimuths.append(Zero())

            speed_expr = thruster_cfg.get('speed', '')
            if isinstance(speed_expr, str) and speed_expr.strip():
                auxiliary_speeds.append(ExcitationSignal1D.from_str(speed_expr, {'seed': seed}))
            else:
                auxiliary_speeds.append(Zero())

        self.auxiliary_cls = auxiliary_azimuths + auxiliary_speeds

        if self.controller is not None:
            initial_commands = self.controller.prev['u']
        if initial_commands is None:
            initial_commands = np.array(6*[0.0])

        super().__init__(initial_commands, seed=seed)


    def __get__(self, states_des:np.ndarray, states:np.ndarray, current:Current, wind:Wind, obstacles:List[Obstacle], target_vessels:List, t: float, *args, **kwargs) -> Tuple[np.ndarray, Dict]:
        aux_commands = np.array([aux_cls(t) for aux_cls in self.auxiliary_cls])
        if self.controller is not None:
            commands, info = self.controller.__get__(states_des, states, current, wind, obstacles, target_vessels, t, *args, **kwargs)
            return commands + aux_commands, info
        return aux_commands, {}
    
    def reset(self, initial_commands: npt.NDArray, seed: Optional[int] = None):
        self.prev = {'u': initial_commands, 'info': None}