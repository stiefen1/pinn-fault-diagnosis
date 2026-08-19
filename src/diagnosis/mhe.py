from python_vehicle_simulator.lib.weather import Wind, Current
from src.diagnosis.base import RevoltFaultDiagnosis

from collections import deque
from typing import Tuple, Dict, Optional
from copy import deepcopy

import casadi as cs, numpy as np

R_MHE = np.diag([1e-2, 1e-2, 0.2*np.pi/180, 5e-2, 5e-2, 5e-2, np.pi/100, np.pi/100])

DEFAULT_IPOPT_SOLVER_OPTS = {
    "error_on_fail": False,
    "expand": True,
    "print_time": False,
    "record_time": True,
    "ipopt.print_level": 0,
    "ipopt.max_iter": 1,
    "ipopt.tol": 1e-6,
    "ipopt.acceptable_tol": 1e-4,
    "ipopt.mu_init": 3e-4, # 1e-3, 3e-4 was great with horizon=80, 1e-2
    "ipopt.warm_start_init_point": "yes",
    "ipopt.alpha_for_y": 'safer-min-dual-infeas',
    "ipopt.mu_strategy": 'monotone'
}

class MHEFaultDiagnosis(RevoltFaultDiagnosis):
    """
    Moving Horizon Estimation for fault diagnosis.

    Decision variables: X ∈ ℝ^(NX×(N+1)), θ ∈ ℝ^NTHETA
    Parameters:         x̄, P̄⁻¹ (arrival cost), Y (measurements), U (inputs), d (disturbance)
    Constraints:        x_{k+1} = f(x_k, u_k, θ, d)
    """

    NX = 18
    NZ = 8
    NU = 6
    NTHETA = 6
    ND = 3
    MEAS_IDX = [0, 1, 5, 6, 7, 11, 12, 13]  # matches RevoltFaultDiagnosis.measurement_model

    def __init__(
            self,
            dt: float,
            horizon: int = 20,
            *args,
            R: np.ndarray = R_MHE,
            P0_inv_diag: Optional[np.ndarray] = None,
            dp_mode: bool = False,
            solver_opts: Optional[dict] = None,
            **kwargs
    ):
        super().__init__(np.zeros(self.NX), dt, *args, dp_mode=dp_mode, **kwargs)
        self.horizon = horizon
        self._R_inv = np.linalg.inv(R)
        self._P0_inv = np.full(self.NX, 1e-4) if P0_inv_diag is None else np.asarray(P0_inv_diag)

        self._buf: deque = deque(maxlen=horizon + 1)  # stores (y, u) pairs
        self._x_hat = np.zeros(self.NX)
        self._theta_hat = np.ones(self.NTHETA)
        self._prev_sol: Optional[tuple] = None

        self._build_nlp(solver_opts)

    def _build_nlp(self, solver_opts: Optional[dict]) -> None:
        N = self.horizon
        fd = self.dynamics._fd          # cs.Function: f(x, u, theta, d) -> x_next (from base class)
        R_inv = cs.DM(self._R_inv)

        X = cs.SX.sym('X', self.NX, N + 1)
        theta = cs.SX.sym('theta', self.NTHETA)
        x_bar = cs.SX.sym('x_bar', self.NX)
        P_inv = cs.SX.sym('P_inv', self.NX)    # diagonal arrival-cost precision
        Y = cs.SX.sym('Y', self.NZ, N + 1)
        U = cs.SX.sym('U', self.NU, N)
        d = cs.SX.sym('d', self.ND)

        # arrival cost + measurement fit
        dx0 = X[:, 0] - x_bar
        obj = cs.dot(dx0, P_inv * dx0)
        for k in range(N + 1):
            ek = Y[:, k] - X[self.MEAS_IDX, k]
            obj += cs.mtimes([ek.T, R_inv, ek])

        # dynamics constraints
        g = [X[:, k + 1] - cs.reshape(fd(X[:, k], U[:, k], theta, d), self.NX, 1) for k in range(N)]

        dec = cs.vertcat(cs.reshape(X, -1, 1), theta)
        p = cs.vertcat(x_bar, P_inv, cs.reshape(Y, -1, 1), cs.reshape(U, -1, 1), d)

        opts = {**DEFAULT_IPOPT_SOLVER_OPTS, **(solver_opts or {})}
        self._solver = cs.nlpsol('mhe', 'ipopt', {'x': dec, 'f': obj, 'g': cs.vertcat(*g), 'p': p}, opts)

        n_xdec = self.NX * (N + 1)
        self._lbx = np.concatenate([np.full(n_xdec, -np.inf), np.zeros(self.NTHETA)])
        self._ubx = np.concatenate([np.full(n_xdec, np.inf), np.ones(self.NTHETA)])
        self._lbg = np.zeros(self.NX * N)
        self._ubg = np.zeros(self.NX * N)
        self._n_xdec = n_xdec

    def __get__(self, states: np.ndarray, control_commands: np.ndarray, measurements: np.ndarray, wind: Wind, current: Current, prev_navigation: Dict, *args, **kwargs) -> Tuple[Dict, Dict]:
        prev_wind = prev_navigation["wind"] if "wind" in prev_navigation.keys() else deepcopy(wind)
        prev_current = prev_navigation["current"] if "current" in prev_navigation.keys() else deepcopy(current)

        self._buf.append((np.asarray(measurements[:self.NZ]), np.asarray(control_commands[:self.NU])))

        if len(self._buf) < self.horizon + 1:
            return {'diagnosis_states': states, 'diagnosis_theta': np.ones(self.NTHETA), 'diagnosis_theta_cov': np.zeros(self.NTHETA)}, {}

        buf = list(self._buf)
        Y_mat = np.column_stack([e[0] for e in buf])           # (NZ, N+1)
        U_mat = np.column_stack([e[1] for e in buf[:-1]])      # (NU, N): drop current u (not yet applied)
        d = self.compute_disturbance(self._x_hat, prev_wind, prev_current)

        p = np.concatenate([
            self._x_hat, self._P0_inv,
            Y_mat.reshape(-1, order='F'),
            U_mat.reshape(-1, order='F'),
            d
        ])

        if self._prev_sol is not None:
            X_prev, th_prev = self._prev_sol
            X_shift = np.hstack([X_prev[:, 1:], X_prev[:, -1:]])
            x0 = np.concatenate([X_shift.reshape(-1, order='F'), th_prev])
        else:
            x0 = np.concatenate([np.tile(self._x_hat, self.horizon + 1), self._theta_hat])

        out = self._solver(x0=x0, lbx=self._lbx, ubx=self._ubx, lbg=self._lbg, ubg=self._ubg, p=p)
        dec = np.array(out['x']).reshape(-1)
        X_opt = dec[:self._n_xdec].reshape(self.NX, self.horizon + 1, order='F')
        theta_opt = np.clip(dec[self._n_xdec:], 0, 1)

        self._x_hat = X_opt[:, -1].copy()
        self._theta_hat = theta_opt.copy()
        self._prev_sol = (X_opt, theta_opt)

        return {
            'diagnosis_states': self._x_hat,
            'diagnosis_theta': theta_opt,
            'diagnosis_theta_cov': np.zeros(self.NTHETA)   # MHE has no natural covariance output
        }, {}