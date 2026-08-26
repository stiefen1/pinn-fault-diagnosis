from python_vehicle_simulator.lib.weather import Wind, Current
from src.diagnosis.base import RevoltFaultDiagnosis, register_diagnosis_module

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
        self.fault_signal = []
        self.control_commands = []
        self.fault_signal_integral = 0
        self.iter_since_detection = 100
        self.corrs = np.array(6*[0])

    def __get__(self, states: np.ndarray, control_commands: np.ndarray, measurements: np.ndarray, wind: Wind, current: Current, prev_navigation: Dict, states_est: np.ndarray, innovation_cov: np.ndarray, *args, **kwargs) -> Tuple[Dict, Dict]:
        prev_wind = prev_navigation["wind_meas"] if "wind_meas" in prev_navigation.keys() else deepcopy(wind)
        prev_current = prev_navigation["current_meas"] if "current_meas" in prev_navigation.keys() else deepcopy(current) 

        if self.iter_since_detection < 100:
            self.iter_since_detection += 1
        else:
            self.corrs = np.array(6*[0])

        self.fault_signal.append(self.fault_indicator(states_est, measurements, innovation_cov))
        if len(self.fault_signal) > self.n_iso:
            self.fault_signal.pop(0)

        self.control_commands.append(control_commands)
        if len(self.control_commands) > self.n_iso:
            self.control_commands.pop(0)

        
        if len(self.control_commands) == len(self.fault_signal) == self.n_iso and not(self.iter_since_detection < 100):
            # corrs, lags = self.isolation(np.array(self.fault_signal), np.array(self.control_commands).T)

            if self.detection(np.array(self.fault_signal)):
                self.corrs = self.isolation(np.array(self.fault_signal), np.array(self.control_commands).T)
                self.iter_since_detection = 0
                


        y = np.asarray(measurements[:self.NZ])
        u = np.asarray(control_commands[:self.NU])
        d = self.compute_disturbance(self.x_hat, prev_wind, prev_current)

        x_pred = self.dynamics.fd(self.x_hat, u, theta=self.theta_hat, disturbance=d).squeeze()
        e      = y - x_pred[self.MEAS_IDX]

        # ∂x_pred/∂θ — same Jacobian used by the EKF; shape (NX, NTHETA)
        Td   = np.array(self.dynamics.Td(self.x_hat, u, self.theta_hat, d))

        W = np.zeros((6, 6))
        if np.sum(np.abs(self.corrs)) > 0:
            idx = np.argmax(self.corrs)
            W[idx, idx] = 1.0
        
        # W = np.abs(np.diag((np.exp(self.corrs/3) / (1 + np.exp(self.corrs/3)) - 0.5) * 2))
        print(np.diag(W))
        grad = W @ Td[self.MEAS_IDX, :].T @ e   # gradient ascent direction, shape (NTHETA,)

        if self.sparse_grad:
            mask = np.zeros_like(grad)
            mask[np.argmax(np.abs(grad))] = 1.0
            grad = grad * mask

        if self.normalize_grad:
            norm = np.linalg.norm(grad)
            if norm > 1e-10:
                grad = grad / norm

        # observed states corrected directly from measurement; unobserved propagated by dynamics
        self.x_hat = x_pred.copy()
        self.x_hat[self.MEAS_IDX] = y

        # print(grad)
        self.theta_hat = np.clip(self.theta_hat + self.lr * grad, 0.0, 1.0)

        
        self.fault_signal_integral += self.fault_signal[-1]

        return {
            'diagnosis_states':    self.x_hat,
            'diagnosis_theta':     self.theta_hat,
            'diagnosis_theta_cov': np.zeros(self.NTHETA),
            'fault_indicator': self.fault_signal[-1],
            'fault_signal_integral': self.fault_signal_integral,
            'corrs': self.corrs
        }, {}

register_diagnosis_module("PEMFaultDiagnosis", PEMFaultDiagnosis)