from python_vehicle_simulator.lib.guidance import IGuidance
from python_vehicle_simulator.lib.path import PWLPath
from python_vehicle_simulator.lib.weather import Current, Wind
from python_vehicle_simulator.lib.obstacle import Obstacle

import numpy as np


from typing import List, Tuple, Dict

class TrajectoryTrackingGuidance(IGuidance):
    def __init__(
            self,
            path: PWLPath,
            desired_speed: float,
            dt: float,
            horizon: int,
            *args,
            **kwargs
    ):
        self.path = path
        self.desired_speed = desired_speed
        self.dt = dt
        self.horizon = horizon
        super().__init__(*args, **kwargs)

    def __get__(self, states: np.ndarray, current:Current, wind:Wind, obstacles:List[Obstacle], target_vessels:List, *args, **kwargs) -> Tuple[np.ndarray, Dict]:
        target_wpts = np.array(self.path.get_target_wpts_from(states[0], states[1], self.desired_speed * self.dt, self.horizon+1))

        trajectory_matrix = np.zeros((self.horizon+1, states.shape[0]))
        trajectory_matrix[:, 0] = target_wpts[:, 0]
        trajectory_matrix[:, 1] = target_wpts[:, 1]
        trajectory_matrix[:, 6] = self.desired_speed

        info = {'term': self.path.progression(states[0], states[1]) >= 1.0}
        return trajectory_matrix, info

    def reset(self):
        pass