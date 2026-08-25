from python_vehicle_simulator.lib.weather import Wind, Current
from src.diagnosis.base import RevoltFaultDiagnosis, register_diagnosis_module
from src.diagnosis.ekf import Revolt3AugmentedEKF, Q_REVOLT_DIAGNOSIS, R_REVOLT_DIAGNOSIS

from typing import Tuple, Dict, Optional
from copy import deepcopy

import numpy as np

R_EKPF_DIAGNOSIS = R_REVOLT_DIAGNOSIS

# Indices in the 18-dim ext_x that correspond to the 10-dim physical state
# [N, E, psi, u, v, r, a1, a2, n1, n2]
_PHYS_IDX = [0, 1, 5, 6, 7, 11, 12, 13, 15, 16]


class EKPFaultDiagnosis(RevoltFaultDiagnosis):
    """
    Extended Kalman Particle Filter (EKPF) for fault parameter estimation.

    Each theta particle carries its own EKF to track the physical state.
    The EKF provides a locally linearised proposal, improving weight quality
    compared to the plain bootstrap PF.

    Theta particles:    [s1, s2, loe1, loe2]    (N, 4)
    Per-particle EKF:   14-dim augmented state   [phys(10), theta(4)]
    """

    def __init__(
            self,
            dt: float,
            n_particles: int,
            *args,
            states: Optional[np.ndarray] = None,
            dp_mode: bool = False,
            theta_process_std: Tuple[float, ...] = (0.002, 0.002, 0.002, 0.002),
            Q: np.ndarray = Q_REVOLT_DIAGNOSIS,
            R: np.ndarray = R_EKPF_DIAGNOSIS,
            P0: Optional[np.ndarray] = None,
            likelihood_temperature: float = 5.0,
            ema_alpha: float = 0.2,
            sparsity_weight: float = 10.0,
            mean_reversion_rate: float = 0.05,
            mean_reversion_exponent: int = 9,
            **kwargs
    ):
        if states is None:
            states = np.array(10 * [0.0] + 4 * [1.0])

        if P0 is None:
            P0 = 1e5 * Q

        self.z_residuals = 0.0

        super().__init__(states, dt, *args, dp_mode=dp_mode, **kwargs)

        self.sparsity_weight = sparsity_weight
        self.mean_reversion_rate = mean_reversion_rate
        self.mean_reversion_exponent = mean_reversion_exponent
        self.n_particles = n_particles
        self.theta_process_std = np.array(theta_process_std)
        self.R_inv = np.linalg.inv(R) / likelihood_temperature
        self.ema_alpha = ema_alpha
        self.weights = np.ones(n_particles) / n_particles
        self._theta_ema: Optional[np.ndarray] = None

        # Initialise one EKF per particle, each slightly perturbed
        theta0 = states[10:14]
        self.ekfs = []
        for i in range(n_particles):
            theta_i = np.clip(
                theta0 + np.random.uniform(-0.1, 0.1, 4),
                0.0, 1.0,
            )
            x_i = np.concatenate([states[:10], theta_i])  # (14,)
            self.ekfs.append(
                Revolt3AugmentedEKF(Q, R, x_i, P0.copy(), dt, dp_mode=dp_mode)
            )

    # --- coordinate helpers (identical to EKF / PF) ---

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
        return z[0:8]

    @staticmethod
    def ext_x_from_self_x(x: np.ndarray) -> np.ndarray:
        return np.concatenate([x[0:2], [0, 0, 0, x[2]], x[3:5], [0, 0, 0, x[5]], x[6:8], [0], x[8:10], [0]])

    @staticmethod
    def self_x_from_ext_x(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        return np.concatenate([x[0:2], x[5:8], x[11:14], x[15:17], theta])

    @staticmethod
    def _systematic_resample(weights: np.ndarray) -> np.ndarray:
        N = len(weights)
        positions = (np.arange(N) + np.random.uniform(0, 1)) / N
        return np.searchsorted(np.cumsum(weights), positions)

    def __get__(
            self,
            states: np.ndarray,
            control_commands: np.ndarray,
            measurements: np.ndarray,
            wind: Wind,
            current: Current,
            prev_navigation: Dict,
            *args,
            **kwargs,
    ) -> Tuple[Dict, Dict]:
        """
        states:           x_k
        control_commands: u_k-1
        measurements:     y_k
        """
        prev_wind = prev_navigation.get("wind_meas", deepcopy(wind))
        prev_current = prev_navigation.get("current_meas", deepcopy(current))

        z = self.self_meas_from_ext_meas(measurements)   # (8,)
        u_self = self.self_u_from_ext_u(control_commands)  # (4,)
        N = self.n_particles

        # Disturbance from weighted-mean physical state (shared approximation)
        x_self_mean = np.zeros(14)
        for i, ekf in enumerate(self.ekfs):
            x_self_mean += self.weights[i] * ekf.x
        ext_mean = self.ext_x_from_self_x(x_self_mean)
        # disturbance = self.compute_disturbance(ext_mean, prev_wind, prev_current)

        log_w = np.zeros(N)
        x_self_particles = np.zeros((N, 14))

        for i, ekf in enumerate(self.ekfs):
            # 1. Perturb theta: component-wise random walk with nonlinear mean-reversion.
            # Drift = κ * θ^p * (1 - θ): peaks at θ* = p/(p+1), so p=9 → peak at θ=0.9.
            # Near-healthy components feel the strongest pull back to 1;
            # genuinely faulty components feel almost no restoring force.
            chosen = np.random.randint(0, 4)
            perturbation = np.zeros(4)
            perturbation[chosen] = np.random.normal(0, self.theta_process_std[chosen])
            theta_i = ekf.x[10:14]
            drift = self.mean_reversion_rate * (theta_i ** self.mean_reversion_exponent) * (1.0 - theta_i)
            proposed = theta_i + drift + perturbation
            proposed = np.where(proposed > 1.0, 2.0 - proposed, proposed)
            proposed = np.where(proposed < 0.0, -proposed, proposed)
            ekf.x[10:14] = np.clip(proposed, 0.0, 1.0)

            # 2. EKF predict step — propagates physical state with current theta
            ekf.predict(u_self, prev_wind, prev_current)

            # 3. Compute likelihood of observation given predicted state
            z_pred = ekf.h(ekf.x)  # (8,)
            residual = z - z_pred
            log_w[i] = -0.5 * residual @ self.R_inv @ residual

            # 4. EKF update step — refines physical state given measurement
            ekf.update(z)

            x_self_particles[i] = ekf.x

        # Sparsity prior: penalise co-occurrence of multiple faults (same as PF).
        # f_i = max(0, 1 - theta_i); penalty = sparsity_weight * sum_{i<j} f_i * f_j
        f = np.maximum(0.0, 1.0 - x_self_particles[:, 10:14])  # (N, 4)
        sum_f = f.sum(axis=1)                                   # (N,)
        log_w -= self.sparsity_weight * 0.5 * (sum_f ** 2 - (f ** 2).sum(axis=1))
        # Normalise weights (SIS)
        log_w -= log_w.max()
        self.weights = np.exp(log_w) * self.weights
        self.weights /= self.weights.sum()

        self.z_residuals = z - x_self_particles[:, :8]  # (N, 8)

        # Weighted mean estimates
        theta_mean = self.weights @ x_self_particles[:, 10:14]   # (4,)
        theta_var  = self.weights @ (x_self_particles[:, 10:14] - theta_mean) ** 2
        x_self_mean = self.weights @ x_self_particles            # (14,)
        x_mean = self.ext_x_from_self_x(x_self_mean)            # (18,)
        self.states = x_self_mean

        # # EMA smoothing on theta output
        # if self._theta_ema is None:
        #     self._theta_ema = theta_mean.copy()
        # else:
        #     self._theta_ema = self.ema_alpha * theta_mean + (1.0 - self.ema_alpha) * self._theta_ema
        # assert self._theta_ema is not None
        # theta_mean = self._theta_ema

        # Resample when ESS drops below N/2
        n_eff = 1.0 / np.sum(self.weights ** 2)
        if n_eff < N / 2:
            idx = self._systematic_resample(self.weights)
            self.ekfs = [deepcopy(self.ekfs[j]) for j in idx]
            self.weights[:] = 1.0 / N

        ext_theta_mean = np.concatenate([theta_mean[0:2], [1.0], theta_mean[2:4], [1.0]])
        ext_theta_var  = np.concatenate([theta_var[0:2],  [0.0], theta_var[2:4],  [0.0]])

        prev_residuals = 0 if self.prev['diagnosis'] is None else self.prev['diagnosis']['residuals']
        prev_pred_error = 0 if self.prev['diagnosis'] is None else self.prev['diagnosis']['prediction_error']

        return {
            'diagnosis_states':    x_mean,
            'diagnosis_theta':     ext_theta_mean,
            'diagnosis_theta_cov': ext_theta_var,
            # 'residuals': self.residuals(x_mean, control_commands, measurements, prev_wind, prev_current, ext_theta_mean) + prev_residuals,
            'prediction_error': self.prediction_error(states, measurements) + prev_pred_error,
        }, {}


register_diagnosis_module("EKPFaultDiagnosis", EKPFaultDiagnosis)