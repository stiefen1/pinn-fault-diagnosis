from python_vehicle_simulator.lib.weather import Wind, Current
from src.diagnosis.base import RevoltFaultDiagnosis

from typing import Tuple, Dict, Optional

import numpy as np

R_PF_DIAGNOSIS = np.diag([1e-2, 1e-2, 0.2*np.pi/180, 5e-2, 5e-2, 5e-2, *np.array(2*[np.pi/100])])

class ParticleFilterFaultDiagnosis(RevoltFaultDiagnosis):
    """
    Bootstrap particle filter for fault parameter (theta) estimation.

    Particles represent the 4 fault parameters [s1, s2, loe1, loe2].
    The physical state is updated as the weighted mean across particles.

    State layout (self_x, 14-dim):
        [N, E, psi, u, v, r, a1, a2, n1, n2, s1, s2, loe1, loe2]
    """

    def __init__(
            self,
            dt: float,
            n_particles: int,
            *args,
            states: Optional[np.ndarray] = None,
            dp_mode: bool = False,
            theta_process_std: float = 0.005,
            R: np.ndarray = R_PF_DIAGNOSIS,
            **kwargs
    ):
        if states is None:
            states = np.array(10*[0.0] + 4*[1.0])

        super().__init__(states, dt, *args, dp_mode=dp_mode, **kwargs)

        self.n_particles = n_particles
        self.theta_process_std = theta_process_std
        self.R_inv = np.linalg.inv(R)

        theta0 = self.self_theta_from_self_x(states)
        self.particles = np.clip(
            theta0 + np.random.uniform(-0.1, 0.1, (n_particles, 4)),
            0, 1
        )
        self.weights = np.ones(n_particles) / n_particles

    # --- coordinate conversions (same as EKF) ---

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

    # --- particle filter steps ---

    @staticmethod
    def _systematic_resample(particles: np.ndarray, weights: np.ndarray) -> np.ndarray:
        N = len(weights)
        positions = (np.arange(N) + np.random.uniform(0, 1)) / N
        indices = np.searchsorted(np.cumsum(weights), positions)
        return particles[indices]

    def __get__(self, states: np.ndarray, control_commands: np.ndarray, measurements: np.ndarray,
                wind: Wind, current: Current, *args, **kwargs) -> Tuple[Dict, Dict]:
        z = self.self_meas_from_ext_meas(measurements)  # (8,)
        x_ext = self.ext_x_from_self_x(self.states)     # (18,) current physical state

        # 1. Predict: random-walk noise on theta particles, clipped to [0, 1]
        self.particles = np.clip(
            self.particles + np.random.normal(0, self.theta_process_std, self.particles.shape),
            0, 1
        )


        # Build batched ext_theta (N, 6): insert fixed 1.0 for the bow thruster
        N = self.n_particles
        ext_thetas = np.hstack([self.particles[:, 0:2], np.ones((N, 1)),
                                self.particles[:, 2:4], np.ones((N, 1))])

        # Propagate physical state with each particle's theta -> (N, 18)
        x_next_batch = self.predict(x_ext, control_commands, wind, current, theta=ext_thetas)

        # 2. Weight: Gaussian likelihood p(z | x_i)
        # Extract predicted measurement [N, E, psi, u, v, r, a1, a2] from ext_x
        z_pred = x_next_batch[:, [0, 1, 5, 6, 7, 11, 12, 13]]  # (N, 8) # [N, E, psi, u, v, r, a1, a2]
        residuals = z - z_pred
        log_w = -0.5 * np.einsum('ni,ij,nj->n', residuals, self.R_inv, residuals)
        log_w -= log_w.max()  # subtract max for numerical stability
        self.weights *= np.exp(log_w)
        self.weights /= self.weights.sum()

        # 3. Estimate: weighted mean and variance
        theta_mean = self.weights @ self.particles
        theta_var  = self.weights @ (self.particles - theta_mean) ** 2
        x_mean     = self.weights @ x_next_batch

        self.states = self.self_x_from_ext_x(x_mean, theta_mean)

        # 4. Resample when effective sample size drops below N/2
        n_eff = 1.0 / np.sum(self.weights ** 2)
        if n_eff < N / 2:
            self.particles = self._systematic_resample(self.particles, self.weights)
            self.weights[:] = 1.0 / N

        ext_theta_mean = np.concatenate([theta_mean[0:2], [1.0], theta_mean[2:4], [1.0]])
        ext_theta_var  = np.concatenate([theta_var[0:2],  [0.0], theta_var[2:4],  [0.0]])

        return {
            'diagnosis_states':    x_mean,
            'diagnosis_theta':     ext_theta_mean,
            'diagnosis_theta_cov': ext_theta_var,
        }, {}

