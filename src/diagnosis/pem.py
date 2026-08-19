from python_vehicle_simulator.lib.weather import Wind, Current
from src.diagnosis.base import RevoltFaultDiagnosis

from typing import Tuple, Dict, Optional
from copy import deepcopy

import numpy as np

class PEMFaultDiagnosis(RevoltFaultDiagnosis):
    """
    Prediction Error Method (PEM) for fault identification
    PEM fault diagnosis: minimise one-step prediction error ||y - h(f(x̂,u,θ̂,d))||²
    w.r.t. θ via online gradient descent.

    Reference: Martinsen et al. (2020), "Combining SysID with RL-based MPC".


    Works well without sensor noise, complete crap with it
    """

    NX = 18
    NZ = 8
    NU = 6
    NTHETA = 6
    MEAS_IDX = [0, 1, 5, 6, 7, 11, 12, 13]  # matches RevoltFaultDiagnosis.measurement_model

    def __init__(
            self,
            dt: float,
            *args,
            lr: float = 8e2,
            dp_mode: bool = False,
            sparse_grad: bool = False,
            normalize_grad: bool = False,
            **kwargs
    ):
        super().__init__(np.zeros(self.NX), dt, *args, dp_mode=dp_mode, **kwargs)
        self.lr = lr
        self.sparse_grad = sparse_grad       # if True, only update the single largest gradient component
        self.normalize_grad = normalize_grad  # decouple step size from Td magnitude (required for practical convergence)
        self.x_hat     = np.zeros(self.NX)
        self.theta_hat = np.ones(self.NTHETA)

    def __get__(self, states: np.ndarray, control_commands: np.ndarray, measurements: np.ndarray, wind: Wind, current: Current, prev_navigation: Dict, *args, **kwargs) -> Tuple[Dict, Dict]:
        prev_wind = prev_navigation["wind"] if "wind" in prev_navigation.keys() else deepcopy(wind)
        prev_current = prev_navigation["current"] if "current" in prev_navigation.keys() else deepcopy(current) 

        y = np.asarray(measurements[:self.NZ])
        u = np.asarray(control_commands[:self.NU])
        d = self.compute_disturbance(self.x_hat, prev_wind, prev_current)

        x_pred = self.dynamics.fd(self.x_hat, u, theta=self.theta_hat, disturbance=d).squeeze()
        e      = y - x_pred[self.MEAS_IDX]

        # ∂x_pred/∂θ — same Jacobian used by the EKF; shape (NX, NTHETA)
        Td   = np.array(self.dynamics.Td(self.x_hat, u, self.theta_hat, d))
        grad = Td[self.MEAS_IDX, :].T @ e   # gradient ascent direction, shape (NTHETA,)
        if self.sparse_grad:
            mask = np.zeros_like(grad)
            mask[np.argmax(np.abs(grad))] = 1.0
            grad = grad * mask

        if self.normalize_grad:
            norm = np.linalg.norm(grad)
            if norm > 1e-10:
                grad = grad / norm

        # observed states corrected directly from measurement; unobserved propagated by dynamics
        self.x_hat              = x_pred.copy()
        self.x_hat[self.MEAS_IDX] = y

        self.theta_hat = np.clip(self.theta_hat + self.lr * grad, 0.0, 1.0)

        return {
            'diagnosis_states':    self.x_hat,
            'diagnosis_theta':     self.theta_hat,
            'diagnosis_theta_cov': np.zeros(self.NTHETA),
        }, {}
