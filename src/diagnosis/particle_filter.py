from python_vehicle_simulator.lib.weather import Wind, Current
from src.diagnosis.base import RevoltFaultDiagnosis

from typing import Tuple, Dict, Optional

import numpy as np

R_PF_DIAGNOSIS = np.diag([1e-2, 1e-2, 0.2*np.pi/180, 5e-2, 5e-2, 5e-2, *np.array(2*[np.pi/100])])

# Indices in the 18-dim ext_x that correspond to the 10-dim physical state
# [N, E, psi, u, v, r, a1, a2, n1, n2]
_PHYS_IDX = [0, 1, 5, 6, 7, 11, 12, 13, 15, 16]


class ParticleFilterFaultDiagnosis(RevoltFaultDiagnosis):
    """
    SMC²-inspired particle filter for fault parameter (theta) estimation.

    Each theta particle carries its own physical state trajectory. This avoids
    the degeneracy of a shared state: a particle with wrong theta accumulates
    prediction errors over time, receiving progressively lower weight without
    needing an explicit horizon buffer.

    Theta particles:            [s1, s2, loe1, loe2]           (N, 4)
    Per-particle physical state [N, E, psi, u, v, r, a1, a2, n1, n2]  (N, 10)
    """

    def __init__(
            self,
            dt: float,
            n_particles: int,
            *args,
            states: Optional[np.ndarray] = None,
            dp_mode: bool = False,
            theta_process_std: Tuple[float, ...] = (0.005, 0.005, 0.005, 0.005),
            R: np.ndarray = R_PF_DIAGNOSIS,
            **kwargs
    ):
        if states is None:
            states = np.array(10*[0.0] + 4*[1.0])

        super().__init__(states, dt, *args, dp_mode=dp_mode, **kwargs)

        self.n_particles = n_particles
        self.theta_process_std = np.array(theta_process_std)
        self.R_inv = np.linalg.inv(R)

        theta0 = self.self_theta_from_self_x(states)
        self.particles = np.clip(
            theta0 + np.random.uniform(-0.1, 0.1, (n_particles, 4)),
            0, 1
        )
        self.weights = np.ones(n_particles) / n_particles

        # Per-particle physical state — the SMC² addition.
        # Each particle evolves its own state so that likelihood information
        # accumulates over time rather than being evaluated from a single mean.
        self.state_estimates = np.tile(states[:10], (n_particles, 1))  # (N, 10)

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

    # --- helpers ---

    def _compute_disturbance(self, ext_state: np.ndarray, wind: Wind, current: Current) -> np.ndarray:
        """Wind + current disturbance force from a single 18-dim ext state."""
        if self.dynamics.vessel_params is None:
            return np.zeros(3)
        p = self.dynamics.vessel_params
        psi, u_vel, v_vel, r = ext_state[5], ext_state[6], ext_state[7], ext_state[11]
        uw, vw = wind.u(psi), wind.v(psi)
        u_rw, v_rw = uw - u_vel, vw - v_vel
        gamma_w = wind.gamma_w(psi)
        wind_rw2 = u_rw**2 + v_rw**2
        tau_coeff = 0.5 * wind.get_air_density() * wind_rw2
        tau_w = np.array([
            tau_coeff * (-p.cx * np.cos(gamma_w)) * p.proj_area_f,
            tau_coeff * ( p.cy * np.sin(gamma_w)) * p.proj_area_l,
            tau_coeff * ( p.cn * np.sin(2 * gamma_w)) * p.proj_area_l * p.loa,
        ])
        uvr = np.array([u_vel, v_vel, r])
        v_c = np.array([current.u(psi), current.v(psi), 0.0])
        tau_c = p.CA(uvr) @ uvr - p.CA(uvr - v_c) @ (uvr - v_c) + p.D @ v_c
        return tau_c + tau_w

    @staticmethod
    def _systematic_resample(weights: np.ndarray) -> np.ndarray:
        """Return resampled indices via systematic resampling."""
        N = len(weights)
        positions = (np.arange(N) + np.random.uniform(0, 1)) / N
        return np.searchsorted(np.cumsum(weights), positions)

    # --- main filter step ---

    def __get__(self, states: np.ndarray, control_commands: np.ndarray, measurements: np.ndarray,
                wind: Wind, current: Current, *args, **kwargs) -> Tuple[Dict, Dict]:
        z = self.self_meas_from_ext_meas(measurements)  # (8,)
        N = self.n_particles

        # 1. Predict: reflected random walk on theta particles
        perturbation = np.random.normal(0, self.theta_process_std, self.particles.shape)
        proposed = self.particles + perturbation
        proposed = np.where(proposed > 1.0, 2.0 - proposed, proposed)
        proposed = np.where(proposed < 0.0, -proposed, proposed)
        self.particles = np.clip(proposed, 0.0, 1.0)  # safety clip for extreme double-bounce

        # Build per-particle ext states (N, 18) from per-particle physical states (N, 10)
        ext_states_batch = np.zeros((N, 18))
        ext_states_batch[:, _PHYS_IDX] = self.state_estimates

        ext_thetas = np.hstack([self.particles[:, 0:2], np.ones((N, 1)),
                                self.particles[:, 2:4], np.ones((N, 1))])  # (N, 6)

        # Disturbance from weighted mean physical state (shared approximation;
        # disturbance is smooth so the mean is a good representative)
        mean_ext = np.zeros(18)
        mean_ext[_PHYS_IDX] = self.weights @ self.state_estimates
        disturbance = self._compute_disturbance(mean_ext, wind, current)  # (3,)

        # Propagate each particle from its own physical state with its own theta
        x_next_batch = self.dynamics.fd_batch(
            ext_states_batch,
            np.tile(control_commands, (N, 1)),   # (N, 6)
            ext_thetas,
            np.tile(disturbance, (N, 1)),         # (N, 3)
        )  # (N, 18)

        # 2. Weight: Gaussian likelihood p(z | x_i)
        z_pred = x_next_batch[:, [0, 1, 5, 6, 7, 11, 12, 13]]  # (N, 8): [N,E,psi,u,v,r,a1,a2]
        residuals = z - z_pred
        log_w = -0.5 * np.einsum('ni,ij,nj->n', residuals, self.R_inv, residuals)
        log_w -= log_w.max()
        self.weights = np.exp(log_w)
        self.weights /= self.weights.sum()

        # 3. Estimate: weighted mean and variance
        theta_mean = self.weights @ self.particles
        theta_var  = self.weights @ (self.particles - theta_mean) ** 2
        x_mean     = self.weights @ x_next_batch

        # 4. Advance each particle's physical state; update shared mean estimate
        self.state_estimates = x_next_batch[:, _PHYS_IDX]  # (N, 10)
        self.states = self.self_x_from_ext_x(x_mean, theta_mean)

        # 5. Resample when ESS drops below N/2; resample state estimates too
        n_eff = 1.0 / np.sum(self.weights ** 2)
        if n_eff < N / 2:
            idx = self._systematic_resample(self.weights)
            self.particles = self.particles[idx]
            self.state_estimates = self.state_estimates[idx]
            self.weights[:] = 1.0 / N

        ext_theta_mean = np.concatenate([theta_mean[0:2], [1.0], theta_mean[2:4], [1.0]])
        ext_theta_var  = np.concatenate([theta_var[0:2],  [0.0], theta_var[2:4],  [0.0]])

        return {
            'diagnosis_states':    x_mean,
            'diagnosis_theta':     ext_theta_mean,
            'diagnosis_theta_cov': ext_theta_var,
        }, {}

