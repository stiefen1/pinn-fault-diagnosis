from python_vehicle_simulator.lib.kalman import IExtendedKalmanFilter
from python_vehicle_simulator.vehicles.revolt3 import ReVolt3Dynamics, RevoltParameters3DOF
import numpy as np

from typing import Optional

class StateEstimatorEKF(IExtendedKalmanFilter):
    """
    Extended Kalman filter to estimate own ship states.

    States:
        Eta: 
            north   [m]
            east    [m]
            down    [m]
            roll    [rad]
            pitch   [rad]
            yaw     [rad]

        Nu: 
            surge speed [m/s]
            sway speed  [m/s]
            heave speed [m/s]
            roll rate   [rad/s]
            pitch rate  [rad/s]
            yaw rate    [rad/s]
        
        Azimuth angles:
            angle 1 [rad]
            angle 2 [rad]
            angle 3 [rad]

        Thruster speeds:
            speed 1 [rad]
            speed 2 [rad]
            speed 3 [rad]
    
    Model: Nonlinear 3DOFs

    """
    vessel_params: RevoltParameters3DOF = RevoltParameters3DOF()

    def __init__(
            self,
            Q, # Process covariance
            R, # Measurement Covariance
            x0, # Initial states
            P0, # Initial error covariance
            dt, # Sampling time, needed when building the system's model.
            dp_mode: bool = True,
            *args,
            **kwargs
        ):
        super().__init__(Q, R, x0, P0, dt, *args, **kwargs)
        self.revolt_dynamics = ReVolt3Dynamics(dt, dp_mode=dp_mode)

    @property
    def dp_mode(self) -> bool:
        return self.revolt_dynamics.dp_mode

    def f(self, x:np.ndarray, u:np.ndarray, *args, theta:Optional[np.ndarray]=None, diagnosis_theta:Optional[np.ndarray]=None, disturbance:Optional[np.ndarray]=None, **kwargs) -> np.ndarray:
        """
        System's model: x' = f(x, u) + v
        """
        if diagnosis_theta is not None:
            theta = diagnosis_theta
        return self.revolt_dynamics.fd(x, u, theta if theta is not None else np.ones((6,)), disturbance if disturbance is not None else np.zeros((3,))).squeeze()
    
    def dfdx(self, x:np.ndarray, u:np.ndarray, *args, theta:Optional[np.ndarray]=None, diagnosis_theta:Optional[np.ndarray]=None, disturbance:Optional[np.ndarray]=None, **kwargs) -> np.ndarray:
        """
        Jacobian of system's model: df/dx for x = x_prev, u = u_prev
        """
        if diagnosis_theta is not None:
            theta = diagnosis_theta
        return self.revolt_dynamics.Ad(x, u, theta if theta is not None else np.ones((6,)), disturbance if disturbance is not None else np.zeros((3,)))
    
    def h(self, x:np.ndarray, *args, **kwargs) -> np.ndarray:
        return np.array([x[0], x[1], x[5], x[6], x[7], x[11], *x[12:15]])

    def dhdx(self, x:np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        Jacobian of the measurement's model: dh/dx for z = h(x)
        """
        return np.array([
            [1] + 17 * [0],                 # North
            [0, 1] + 16 * [0],              # East
            5 * [0] + [1] + 12 * [0],       # yaw
            6 * [0] + [1] + 11 * [0],       # surge speed
            7 * [0] + [1] + 10 * [0],       # sway speed
            11 * [0] + [1] + 6 * [0],       # yaw rate
            12 * [0] + [1] + 5 * [0],       # azimuth 1
            13 * [0] + [1] + 4 * [0],       # azimuth 2
            14 * [0] + [1] + 3 * [0],       # azimuth 3
        ])

    
if __name__ == "__main__":
    from python_vehicle_simulator.lib.weather import Wind, Current
    Q = np.eye(18, 18)
    R = np.eye(9, 9)
    x0 = np.ones((18,)) * 0.1
    P0 = np.ones((18, 18)) * 0.1
    dt = 0.2

    state_estimator = StateEstimatorEKF(
        Q,
        R,
        x0,
        P0,
        dt
    )


    print(state_estimator.predict(np.array(6*[0]),Wind(0, 0), Current(0, 0)))