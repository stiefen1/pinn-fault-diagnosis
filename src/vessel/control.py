from python_vehicle_simulator.nmpc.tracker import NMPCTracker
from python_vehicle_simulator.vehicles.revolt3 import RevoltParameters3DOF, RevoltThrusterParameters, ReVolt3Dynamics

from typing import Literal, Optional, Dict

import numpy as np, numpy.typing as npt, casadi as cs

class NMPCTrajectoryTrackerRevolt(NMPCTracker):
    NX = 18
    NU = 6
    NTHETA = 6 # LOE + Azimuth stucked
    NDISTURBANCES = 3 # 3DOFs generalized forces tau_u, tau_v, tau_r
    STATES_TO_TRACK_IDX = [0, 1, 6, 7]

    thruster_params: RevoltThrusterParameters = RevoltThrusterParameters()
    vessel_params: RevoltParameters3DOF = RevoltParameters3DOF()

    def __init__(
            self,
            horizon: int,
            dt: float,
            dp_mode: bool = True,
            solver: Literal["ipopt"] = "ipopt",
            solver_opts: Optional[Dict] = None,
            u_0: Optional[npt.NDArray] = None, # previous control command
            seed: Optional[int] = None,
            Q: Optional[npt.NDArray] = np.diag([1, 1, 1, 0.1]),
            R: Optional[npt.NDArray] = np.diag(3*[1e-6] + 3*[1e-5]),
            QN: Optional[npt.NDArray] = None, # Equal to Q if None
            singularity_eps: float = 1e-6,
            singularity_weight: float = 1e-5,
            **kwargs
    ):
        self.dp_mode = dp_mode
        alpha_lb = self.thruster_params.alpha_min.copy()
        alpha_ub = self.thruster_params.alpha_max.copy()
        speed_lb = self.thruster_params.speed_min.copy()
        speed_ub = self.thruster_params.speed_max.copy()
        self.singularity_eps = singularity_eps
        self.singularity_weight = singularity_weight

        if not self.dp_mode:
            # Disable bow commands in optimization when DP mode is off.
            alpha_lb[2] = 0.0
            alpha_ub[2] = 0.0
            speed_lb[2] = 0.0
            speed_ub[2] = 0.0

        super().__init__(
            horizon,
            ReVolt3Dynamics(dt, dp_mode=dp_mode)._fd,
            self.NX,
            self.NU,
            self.NTHETA,
            self.NDISTURBANCES,
            u_lb=np.concatenate([alpha_lb, speed_lb]),
            u_ub=np.concatenate([alpha_ub, speed_ub]),
            solver=solver,
            solver_opts=solver_opts,
            u_0=u_0,
            seed=seed,
            Q=Q,
            R=R,
            QN=QN
        )

    def lagrange(self, xk: cs.SX, uk: cs.SX, x_ref_k: cs.SX, k: int) -> cs.SX:
        e = self._tracking_error(xk, x_ref_k)
        B = self.thruster_params.Alpha(xk[12], xk[13], xk[14])
        if not(self.dp_mode):
            B = B[0:2, 0:2]
        return cs.mtimes([e.T, self.Q, e]) + cs.mtimes([uk.T, self.R, uk]) + self.singularity_weight/(cs.det(B@B.T)+self.singularity_eps)

    def mayer(self, xN: cs.SX, x_ref_N: cs.SX) -> cs.SX:
        eN = self._tracking_error(xN, x_ref_N)
        B = self.thruster_params.Alpha(xN[12], xN[13], xN[14])
        if not(self.dp_mode):
            B = B[0:2, 0:2]
        return cs.mtimes([eN.T, self.QN, eN]) + self.singularity_weight/(cs.det(B@B.T)+self.singularity_eps)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    controller = NMPCTrajectoryTrackerRevolt(
        30,
        0.2,
        Q=np.diag([1.0, 1.0, 0.1, 0.1]),
        R=np.diag(3*[1e-6] + 3*[1e-5])
    )
    u_ref = 0.5
    x_des = np.array([1.0, 0.5] + 4*[0.0] + [u_ref] + 11 * [0.0]) # these are conflicting objectives over the horizon
    x = np.array(18*[0.0])
    u0, info = controller(x_des, x, None, None, [], [])

    print(u0)
    print(info['x_pred'].shape)
    print(info['stats']['return_status'], info['stats']['success'])

    fig, ax = plt.subplots()
    ax.plot(info['x_pred'][:, 1], info['x_pred'][:, 0])
    ax.set_xlim(-1, 1)
    ax.set_aspect('equal')
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(info['x_pred'][:, 6])
    plt.show()