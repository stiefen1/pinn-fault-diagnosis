from python_vehicle_simulator.lib.navigation import INavigation
from python_vehicle_simulator.lib.obstacle import Obstacle
from python_vehicle_simulator.lib.sensor import ISensor
from python_vehicle_simulator.lib.weather import Wind, Current
from python_vehicle_simulator.vehicles.vessel import IVessel
from python_vehicle_simulator.vehicles.revolt3 import  RevoltParameters3DOF

from src.vessel.state_estimator import StateEstimatorEKF

from matplotlib.axes import Axes
from datetime import datetime
from typing import List, Optional, Tuple, Dict

import numpy as np, numpy.typing as npt, gymnasium as gym


Q_REVOLT = np.diag([0.3**2, 0.3**2, 0, 0, 0, (0.4*np.pi/180)**2, 0.02**2, 0.02**2, 0, 0, 0, 15*np.pi/180/3600, *np.array(3*[np.pi/100]), *np.array(3*[1.0])]) / 100
R_REVOLT = np.diag([1e-2, 1e-2, 0.2*np.pi/180, 5e-2, 5e-2, 5e-2, *np.array(3*[np.pi/100])])

R_WIND = np.diag([np.pi/5, 0.2])    # direction [rad], speed [m/s]
R_CURRENT = np.diag([np.pi/20, 0.1]) # direction [rad], speed [m/s]

class NavigationRevolt(INavigation):
    """
        According to https://ntnuopen.ntnu.no/ntnu-xmlui/handle/11250/2452115, measurement uncertainties for ReVolt are:

        heading +- 0.2°
        position +- 1cm
        u, v +- 0.05 m/s
        r not specified, assuming it is very low according to graph. let's say r +- 0.05 deg/s as well
    """
    def __init__(
            self,
            states: np.ndarray,
            dt: float,
            *args,
            dp_mode: bool = True,
            Q_se: Optional[np.ndarray] = Q_REVOLT,          # process noise     (state estimator)
            R_se: Optional[np.ndarray] = R_REVOLT,          # measurement noise (state estimator)
            P0_se: np.ndarray = np.eye(18),                 # state covariance  (state estimator)
            seed: Optional[int] = None,
            sensors: Dict[str, ISensor] = {},
            vessel_params: RevoltParameters3DOF = RevoltParameters3DOF(),
            perfect_meas: bool = False,
            **kwargs
    ):
        self.vessel_params = vessel_params
        self.perfect_meas = perfect_meas

        self.state_estimator_params = {
            'Q': Q_se,
            'R': R_se,
            'P0': P0_se,
            'dt': dt
        }

        self.state_estimator = StateEstimatorEKF(
            **self.state_estimator_params,
            x0=states,
            dp_mode=dp_mode,
        )
        # self.dp_mode = dp_mode
        
        super().__init__(states, sensors, *args, **kwargs)
        self.reset(states, seed=seed)

    @property
    def dp_mode(self) -> bool:
        return self.state_estimator.dp_mode

    def measure_states(self, states: npt.NDArray) -> npt.NDArray:
        noise = self.np_random.multivariate_normal(np.array(9*[0]), R_REVOLT) * (not(self.perfect_meas))
        noisy_states = np.array([states[0], states[1], states[5], states[6], states[7], states[11], *states[12:15]]) + noise
        return noisy_states

    def measure_wind(self, wind: Wind) -> Wind:
        if self.perfect_meas:
            beta, norm = wind._beta, wind._norm
        else:
            beta, norm = wind._beta_0, wind._norm_0 # assume measurement is just a constant value, e.g. no sensors onboard but access to a low-freq API 
        return Wind(beta, norm) # EXTREMELY IMPORTANT TO CREATE A NEW OBJECT -> OTHERWISE INITIAL OBJECT WILL BE AFFECTED (MEMORY IS SHARED) 
    
    def measure_current(self, current: Current) -> Current:
        if self.perfect_meas:
            beta, norm = current._beta, current._norm
        else:
            beta, norm = current._beta_0, current._norm_0 # assume measurement is just a constant value, e.g. no sensors onboard but access to a low-freq API
        return Current(beta, norm) # EXTREMELY IMPORTANT TO CREATE A NEW OBJECT -> OTHERWISE INITIAL OBJECT WILL BE AFFECTED (MEMORY IS SHARED)
        
    def __get__(self, states:np.ndarray, current:Current, wind:Wind, obstacles:List[Obstacle], target_vessels:List[IVessel],  control_commands: np.ndarray, *args, timestamp: Optional[datetime] = None, theta:Optional[np.ndarray]=None, **kwargs) -> Tuple[Dict, Dict]:
        """
        target_vessels are does that are part of the simulation, i.e that we have control over.
        AIS is considered as a Sensor, and hence is and instance of ISensor.
        """
        # print(timestamp)
        wind = wind if wind is not None else Wind(0, 0)
        current = current if current is not None else Current(0, 0)
        wind_meas = self.measure_wind(wind)
        current_meas = self.measure_current(current)

        states_meas = self.measure_states(states)
        states_est = self.state_estimator(control_commands, states_meas, wind_meas, current_meas, theta=theta)
        
        observation = {
            "eta": states_est[0:6],
            "nu": states_est[6:12],
            "states": states,
            "states_est": states_est,
            "current_meas": current_meas,
            "wind_meas": wind_meas,
            "current": current,
            "wind": wind,
            "obstacles": obstacles,
            "measurements": states_meas,
            "target_vessels": target_vessels, # Required for IGuidance
            "theta": theta if theta is not None else np.ones((6,)),
            "innovation_cov": self.state_estimator.S.copy()
        }
        info = {}

        return observation, info
    
    def reset(self, states: npt.NDArray, seed: Optional[int] = None):
        self.prev = {"eta": states[0:6].copy(), "nu": states[6:12].copy(), "states": states.copy(), "current": None, "wind": None, "obstacles": None, "target_vessels": None, 'info': None}
        self.np_random, _ = gym.utils.seeding.np_random(seed) # type: ignore
        self.state_estimator.reset(states)

    def __plot__(self, ax:Axes, *args, verbose:int=0, **kwargs) -> Axes:
        if self.last_observation is None:
            return ax
        
        eta = self.last_observation["eta"]

        if verbose >= 1:
            x, y = eta[1], eta[0]  # east, north
            
            # Plot the vessel position
            ax.scatter(x, y, c='purple', marker='x')
        
        if verbose >= 5:
            x, y, psi = eta[1], eta[0], eta[5]  # heading in radians
            # Plot an arrow showing the heading direction
            arrow_length = 10  # Adjust as needed for visualization
            dx = arrow_length * np.sin(psi)  # East component
            dy = arrow_length * np.cos(psi)  # North component
            
            ax.arrow(x, y, dx, dy, head_width=2, head_length=3, fc='purple', ec='purple')

        return ax

    def __scatter__(self, ax:Axes, *args, **kwargs) -> Axes:
        if self.last_observation is None:
            return ax
        
        eta = self.last_observation["eta"]
        ax.scatter(eta[1], eta[0], c='purple')
        return ax

    def __fill__(self, ax:Axes, *args, **kwargs) -> Axes:
        return ax
    
if __name__ == "__main__":
    nav = NavigationRevolt(np.array(18*[0]), 0.2)
    print(nav(np.array(18*[0.1]), None, None, [], [], timestamp=datetime.now(), control_commands=np.array(6*[1])))
    