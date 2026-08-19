from python_vehicle_simulator.lib.weather import Wind, Current
from src.diagnosis.base import RevoltFaultDiagnosis

from typing import Tuple, Dict, Optional
from copy import deepcopy

import numpy as np

Q_REVOLT_DIAGNOSIS = np.diag([0.3**2, 0.3**2, (0.4*np.pi/180)**2, 0.02**2, 0.02**2, 15*np.pi/180/3600, *np.array(2*[np.pi/100]), *np.array(2*[1.0]), *np.array(2*[2e-3]), *np.array(2*[2e-3])]) / 100
R_REVOLT_DIAGNOSIS = np.diag([1e-2, 1e-2, 0.2*np.pi/180, 5e-2, 5e-2, 5e-2, *np.array(2*[np.pi/100])])


class UKFFaultDiagnosis(RevoltFaultDiagnosis):
    """
    Unscented Kalman Filter fault diagnosis with augmented state:

    N, E, psi, u, v, r, a1, a2, n1, n2, s1, s2, loe1, loe2
    """

    def __init__(
            self,
            dt: float,
            *args,
            P0: np.ndarray = 10*Q_REVOLT_DIAGNOSIS,
            Q: np.ndarray = Q_REVOLT_DIAGNOSIS,
            R: np.ndarray = R_REVOLT_DIAGNOSIS,
            states: Optional[np.ndarray] = None,
            dp_mode: bool = False,
            frozen_states: Optional[np.ndarray] = None,
            alpha: float = 5e-1,
            beta: float = 2.0,
            kappa: float = 0.0,
            **kwargs
    ):
        if states is None:
            states = np.array(10*[0.0] + 4*[1.0])

        super().__init__(states, dt, *args, dp_mode=dp_mode, **kwargs)

        self.Q = Q.copy()
        self.R = R.copy()
        self.x = states.copy()
        self.P0 = P0.copy()
        self.P = P0.copy()
        self.frozen_states = np.asarray(frozen_states, dtype=int) if frozen_states is not None else np.array([], dtype=int)

        n = len(states)
        lam = alpha**2 * (n + kappa) - n
        self._lam = lam
        self._n = n
        self.Wm = np.full(2*n + 1, 1.0 / (2*(n + lam)))
        self.Wc = np.full(2*n + 1, 1.0 / (2*(n + lam)))
        self.Wm[0] = lam / (n + lam)
        self.Wc[0] = lam / (n + lam) + (1 - alpha**2 + beta)

    # --- same static helpers as EKF ---

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
        return z[0:8]

    @staticmethod
    def ext_x_from_self_x(x: np.ndarray) -> np.ndarray:
        return np.concatenate([x[0:2], [0, 0, 0, x[2]], x[3:5], [0, 0, 0, x[5]], x[6:8], [0], x[8:10], [0]])

    @staticmethod
    def self_x_from_ext_x(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        return np.concatenate([x[0:2], x[5:8], x[11:14], x[15:17], theta])

    def _f(self, x: np.ndarray, u: np.ndarray, disturbance: np.ndarray) -> np.ndarray:
        x_ext = self.ext_x_from_self_x(x)
        x_next = self.dynamics.fd(x_ext, self.ext_u_from_self_u(u), self.ext_theta_from_self_x(x), disturbance).squeeze()
        theta_next = np.clip(x[10:14], 0, 1)
        x_out = self.self_x_from_ext_x(x_next, theta_next)
        x_out[self.frozen_states] = x[self.frozen_states]
        return x_out

    def _h(self, x: np.ndarray) -> np.ndarray:
        return x[0:8]

    def _sigma_points(self) -> np.ndarray:
        try:
            L = np.linalg.cholesky((self._n + self._lam) * self.P)
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky((self._n + self._lam) * (self.P + 1e-8 * np.eye(self._n)))
        sigmas = np.empty((2*self._n + 1, self._n))
        sigmas[0] = self.x
        for i in range(self._n):
            sigmas[i + 1] = self.x + L[:, i]
            sigmas[self._n + i + 1] = self.x - L[:, i]
        return sigmas

    def _predict(self, u: np.ndarray, wind: Wind, current: Current) -> None:
        disturbance = self.compute_disturbance(self.ext_x_from_self_x(self.x), wind, current)
        sigmas = self._sigma_points()
        sigmas_f = np.array([self._f(s, u, disturbance) for s in sigmas])
        self.x = (self.Wm[:, None] * sigmas_f).sum(axis=0)
        diff = sigmas_f - self.x
        self.P = (self.Wc[:, None, None] * np.einsum('ki,kj->kij', diff, diff)).sum(axis=0) + self.Q
        self._sigmas_f = sigmas_f

    def _update(self, z: np.ndarray) -> np.ndarray:
        sigmas_h = np.array([self._h(s) for s in self._sigmas_f])
        z_pred = (self.Wm[:, None] * sigmas_h).sum(axis=0)
        diff_x = self._sigmas_f - self.x
        diff_z = sigmas_h - z_pred
        S = (self.Wc[:, None, None] * np.einsum('ki,kj->kij', diff_z, diff_z)).sum(axis=0) + self.R
        Pxz = (self.Wc[:, None, None] * np.einsum('ki,kj->kij', diff_x, diff_z)).sum(axis=0)
        K = Pxz @ np.linalg.pinv(S)
        K[self.frozen_states, :] = 0
        self.x = self.x + K @ (z - z_pred)
        self.x[10:14] = np.clip(self.x[10:14], 0, 1)
        self.P = self.P - K @ S @ K.T
        return self.x

    def __get__(self, states: np.ndarray, control_commands: np.ndarray, measurements: np.ndarray, wind: Wind, current: Current, prev_navigation: Dict, *args, **kwargs) -> Tuple[Dict, Dict]:
        prev_wind = prev_navigation["wind"] if "wind" in prev_navigation.keys() else deepcopy(wind)
        prev_current = prev_navigation["current"] if "current" in prev_navigation.keys() else deepcopy(current)
        self._predict(self.self_u_from_ext_u(control_commands), prev_wind, prev_current)
        x = self._update(self.self_meas_from_ext_meas(measurements))
        return {
            'diagnosis_states': self.ext_x_from_self_x(x),
            'diagnosis_theta': self.ext_theta_from_self_x(x),
            'diagnosis_theta_cov': self.ext_theta_from_self_x(np.diag(self.P))
        }, {}
