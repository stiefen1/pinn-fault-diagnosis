from python_vehicle_simulator.lib.weather import Wind, Current
from src.diagnosis.base import RevoltFaultDiagnosis

from typing import Tuple, Dict, Optional

import numpy as np

# Measurement indices matching RevoltFaultDiagnosis.measurement_model
_MEAS_IDX = [0, 1, 5, 6, 7, 11, 12, 13]  # N, E, psi, u, v, r, alpha1, alpha2


def _default_L(nx: int = 18, nz: int = 8) -> np.ndarray:
    """Proportional observer gain: correct measured states directly."""
    L = np.zeros((nx, nz))
    for col, row in enumerate(_MEAS_IDX):
        L[row, col] = 0.5
    return L


def _default_Gamma(ntheta: int = 6, nz: int = 8) -> np.ndarray:
    """
    Fault adaptation gain.  Measurement layout (y-indices):
      3=u (surge), 4=v (sway), 5=r (yaw rate), 6=alpha1, 7=alpha2
    theta layout: [az_stuck_0, az_stuck_1, az_stuck_bow, loe_0, loe_1, loe_bow]
    Port/starboard pairs use opposite signs on yaw rate to differentiate lateral faults.
    alpha1/alpha2 are coupled directly to their respective azimuth-stuck parameters.
    Bow entries (2, 5) are left at zero — not estimated in non-DP mode.
    """
    G = np.zeros((ntheta, nz))
    # theta[0:3] = azimuth stuck, theta[3:6] = LOE
    # y-index map: 3=u, 4=v, 5=r, 6=alpha1, 7=alpha2
    # Port stern (thruster 0): positive yaw coupling; alpha1 directly signals a stuck azimuth
    G[0, 3] =  1e-1;  G[0, 4] =  1e-3;  G[0, 5] =  1e-1;  G[0, 6] = 1e-1   # az stuck 0
    # Starboard stern (thruster 1): opposite yaw/sway coupling; alpha2 for stuck
    G[1, 3] =  1e-1;  G[1, 4] = -1e-3;  G[1, 5] = -1e-1;  G[1, 7] = 1e-1   # az stuck 1
    # G[2] = 0 — bow azimuth, not estimated
    # LOE does not affect azimuth dynamics, so alpha columns stay zero
    G[3, 3] =  1e-1;                     G[3, 5] =  1e-1                     # LOE 0 (port)
    G[4, 3] =  1e-1;                     G[4, 5] = -1e-1                     # LOE 1 (starboard)
    # G[5] = 0 — bow LOE, not estimated
    return G


class SMOFaultDiagnosis(RevoltFaultDiagnosis):
    """
    Sliding Mode Observer for fault estimation.

    Predict:    x̂_pred  = f(x̂, u, θ̂, d)
    Error:      e        = y − h(x̂_pred)
    State:      x̂       ← x̂_pred + L @ e
    Theta:      θ̂       ← clip(θ̂ + dt · Γ @ sat(e/δ), 0, 1)
    """

    NX = 18
    NZ = 8
    NU = 6
    NTHETA = 6
    MEAS_IDX = _MEAS_IDX

    def __init__(
            self,
            dt: float,
            *args,
            L: Optional[np.ndarray] = None,
            Gamma: Optional[np.ndarray] = None,
            delta: float = 0.05,
            dp_mode: bool = False,
            **kwargs
    ):
        super().__init__(np.zeros(self.NX), dt, *args, dp_mode=dp_mode, **kwargs)

        self.L     = L     if L     is not None else _default_L(self.NX, self.NZ)
        self.Gamma = Gamma if Gamma is not None else _default_Gamma(self.NTHETA, self.NZ)
        self.delta = delta

        self.x_hat     = np.zeros(self.NX)
        self.theta_hat = np.ones(self.NTHETA)

    def _sat(self, e: np.ndarray) -> np.ndarray:
        return np.clip(e / self.delta, -1.0, 1.0)

    def __get__(self, states: np.ndarray, control_commands: np.ndarray, measurements: np.ndarray, wind: Wind, current: Current, *args, **kwargs) -> Tuple[Dict, Dict]:
        y = np.asarray(measurements[:self.NZ])
        u = np.asarray(control_commands[:self.NU])
        d = self.compute_disturbance(self.x_hat, wind, current)

        x_pred = self.dynamics.fd(self.x_hat, u, theta=self.theta_hat, disturbance=d).squeeze()

        e = y - x_pred[self.MEAS_IDX]

        self.x_hat     = x_pred + self.L @ e
        self.theta_hat = np.clip(self.theta_hat + self.dynamics.dt * self.Gamma @ self._sat(e), 0.0, 1.0)

        return {
            'diagnosis_states':    self.x_hat,
            'diagnosis_theta':     self.theta_hat,
            'diagnosis_theta_cov': np.zeros(self.NTHETA),
        }, {}
