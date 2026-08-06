from python_vehicle_simulator.vehicles.revolt3 import ReVolt3Dynamics, RevoltThrusterParameters, RevoltParameters3DOF
from python_vehicle_simulator.lib.kalman import IExtendedKalmanFilter
from python_vehicle_simulator.lib.weather import Wind, Current
from src.diagnosis.base import RevoltFaultDiagnosis

from typing import Tuple, Dict, Optional

import numpy as np

Q_REVOLT_DIAGNOSIS = np.diag([0.3**2, 0.3**2, (0.4*np.pi/180)**2, 0.02**2, 0.02**2, 15*np.pi/180/3600, *np.array(2*[np.pi/100]), *np.array(2*[1.0]), *np.array(2*[1e-5]), *np.array(2*[1e-5])]) / 100
R_REVOLT_DIAGNOSIS = np.diag([1e-2, 1e-2, 0.2*np.pi/180, 5e-2, 5e-2, 5e-2, *np.array(2*[np.pi/100])])

class Revolt3AugmentedEKF(IExtendedKalmanFilter):
    vessel_params: RevoltParameters3DOF = RevoltParameters3DOF()

    """
    Augmented state-space is

    N, 
    E,
    psi,
    u,
    v,
    r,
    a1,
    a2,
    n1,
    n2,
    s1,
    s2,
    loe1,
    loe2,

    """

    def __init__(
            self,
            Q, # Process covariance
            R, # Measurement Covariance
            x0, # Initial states: [x, theta]
            P0, # Initial error covariance
            dt, # Sampling time, needed when building the system's model.
            dp_mode: bool = True,
            frozen_states: Optional[np.ndarray] = None,
            *args,
            **kwargs
        ):
        super().__init__(Q, R, x0, P0, dt, *args, **kwargs)
        self.revolt_dynamics = ReVolt3Dynamics(dt, dp_mode=dp_mode)
        self._static_frozen: np.ndarray = np.asarray(frozen_states, dtype=int) if frozen_states is not None else np.array([], dtype=int)
        self.frozen_states: np.ndarray = self._static_frozen.copy()

    @property
    def dp_mode(self) -> bool:
        return self.revolt_dynamics.dp_mode
    
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

    def f(self, x:np.ndarray, u:np.ndarray, *args, disturbance:Optional[np.ndarray]=None, **kwargs) -> np.ndarray:
        """
        System's model: x' = f(x, u) + v
        """        
        x_zero_dp_mode = self.ext_x_from_self_x(x)
        x_next = self.revolt_dynamics.fd(x_zero_dp_mode, self.ext_u_from_self_u(u), self.ext_theta_from_self_x(x), disturbance if disturbance is not None else np.zeros((3,))).squeeze()
        theta_next = np.clip(x[10:14], 0, 1) # fault parameters assumed constant
        x_out = self.self_x_from_ext_x(x_next, theta_next)
        x_out[self.frozen_states] = x[self.frozen_states]
        return x_out
    
    def dfdx(self, x:np.ndarray, u:np.ndarray, *args, disturbance:Optional[np.ndarray]=None, **kwargs) -> np.ndarray:
        """
        Jacobian of system's model: df/dx for x = x_prev, u = u_prev
        """
        dfdx = self.revolt_dynamics.Ad(self.ext_x_from_self_x(x), self.ext_u_from_self_u(u), self.ext_theta_from_self_x(x), disturbance if disturbance is not None else np.zeros((3,)))
        dfdtheta = self.revolt_dynamics.Td(self.ext_x_from_self_x(x), self.ext_u_from_self_u(u), self.ext_theta_from_self_x(x), disturbance if disturbance is not None else np.zeros((3,)))
        full_dfdx = np.block([[dfdx, dfdtheta], [np.zeros((6, 18)), np.eye(6)]]) # \in (nx + nt x nx + nt)
        return np.delete(np.delete(full_dfdx, [2, 3, 4, 8, 9, 10, 14, 17, 20, 23], axis=0), [2, 3, 4, 8, 9, 10, 14, 17, 20, 23], axis=1)
    
    def h(self, x:np.ndarray, *args, **kwargs) -> np.ndarray:
        return x[0:8]

    def dhdx(self, x:np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        Jacobian of the measurement's model: dh/dx for z = h(x)
        """
        return np.block([
            [np.eye(8, 8), np.zeros((8, 6))]
        ])

    def update(self, z: np.ndarray) -> np.ndarray:
        dHdx = self.dhdx(self.x)
        S = dHdx @ self.P @ dHdx.T + self.R
        K = self.P @ dHdx.T @ np.linalg.pinv(S)
        K[self.frozen_states, :] = 0
        y = z - self.h(self.x)
        self.x = self.x + K @ y
        self.x[10:14] = np.clip(self.x[10:14], 0, 1)
        self.P = (np.eye(self.P.shape[0]) - K @ dHdx) @ self.P
        return self.x

class EKFFaultDiagnosis(RevoltFaultDiagnosis):
    def __init__(
            self,
            dt: float,
            *args,
            P0: np.ndarray = 1e5*Q_REVOLT_DIAGNOSIS,
            Q: np.ndarray = Q_REVOLT_DIAGNOSIS,
            R: np.ndarray = R_REVOLT_DIAGNOSIS,
            states: Optional[np.ndarray] = None,
            dp_mode: bool = False,
            frozen_states: Optional[np.ndarray] = None,
            **kwargs
    ):
        if states is None:
            states = np.array(10*[0.0] + 4*[1.0])

        super().__init__(states, dt, *args, dp_mode=dp_mode, **kwargs)
        self.ekf = Revolt3AugmentedEKF(Q, R, states, P0, dt, dp_mode=dp_mode, frozen_states=frozen_states)

    def __get__(self, states:np.ndarray, control_commands:np.ndarray, measurements:np.ndarray, wind: Wind, current: Current, *args, **kwargs) -> Tuple[Dict, Dict]:
        x = self.ekf(self.ekf.self_u_from_ext_u(control_commands), self.ekf.self_meas_from_ext_meas(measurements), wind, current)
        return {
            'diagnosis_states': self.ekf.ext_x_from_self_x(x),
            'diagnosis_theta': self.ekf.ext_theta_from_self_x(x),
            'diagnosis_theta_cov': self.ekf.ext_theta_from_self_x(np.diag(self.ekf.P))
        }, {}