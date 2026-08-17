from python_vehicle_simulator.lib.simulator import Simulator
from python_vehicle_simulator.lib.env import NavEnv
from python_vehicle_simulator.vehicles.revolt3 import ReVolt3

from python_vehicle_simulator.lib.weather import Wind, Current
from python_vehicle_simulator.utils.unit_conversion import DEG2RAD
from python_vehicle_simulator.lib.path import PWLPath

from src.vessel.guidance import TrajectoryTrackingGuidance
from src.vessel.navigation import NavigationRevolt
from src.vessel.control import NMPCTrajectoryTrackerRevolt

from src.diagnosis.ekf import EKFFaultDiagnosis
from src.diagnosis.particle_filter import ParticleFilterFaultDiagnosis
from src.diagnosis.parity_space import ParitySpaceFaultDiagnosis
from src.diagnosis.ekpf import EKPFaultDiagnosis

import numpy as np, matplotlib.pyplot as plt

dt = 0.2
horizon = 30
dp_mode = False

vessel = ReVolt3(
        dt,
        dp_mode=dp_mode,
        control=NMPCTrajectoryTrackerRevolt(
            horizon,
            dt,
            dp_mode=dp_mode,
            # singularity_weight=3e-4
        ),
        guidance=TrajectoryTrackingGuidance(
            PWLPath.sample(d_tot=500, max_turn_deg=90, seg_len_range=(3, 5), seed=42).smooth(3),
            0.5,
            dt,
            horizon            
        ),
        navigation=NavigationRevolt(np.array(18*[0]), dt, dp_mode=dp_mode, perfect_meas=False),
        # diagnosis=EKPFaultDiagnosis(
        #     dt,
        #     n_particles=200,
        #     theta_process_std=(0.01, 0.01, 0.01, 0.01)
        # )
        diagnosis=ParticleFilterFaultDiagnosis(
            dt,
            n_particles=500,
            theta_process_std=(0.01, 0.01, 0.01, 0.01)
        )
        # diagnosis=EKFFaultDiagnosis(
        #     dt, 
        #     dp_mode=dp_mode,
        #     frozen_states=np.array([10, 11])
        # )
    )

env = NavEnv(
    own_vessel=vessel,
    target_vessels=[],
    obstacles=[],
    dt=dt,
    current=Current(beta=-30.0*DEG2RAD, v=0.1),
    wind=Wind(beta=60.0*DEG2RAD, v=10.0)
)

sim = Simulator(
        env,
        dt=dt,
        render_mode="human",
        verbose=2,
        skip_frames=1,
        window_size=(6, 6)
    )

sim.run(tf=100, render=True, store_data=True, theta=np.array([1, 1, 1, 0.5, 1, 1]))

# After calling plot_gnc_data_multi, add:
nav_data = sim.simulation_data['gnc_data']['navigation']
vessel_data = sim.simulation_data['own_vessel_states']

fig1 = sim.plot_gnc_data_multi([
    'navigation.eta[0]',
    'vessel.eta[0]'
    ], x_path=['navigation.eta[1]', 'vessel.eta[1]'])
vessel.guidance.path.plot(ax=fig1.axes[0])
fig1.axes[0].set_aspect('equal')

fig5 = sim.plot_gnc_data_multi([
    'vessel.nu[0]', 'vessel.nu[1]'
])

fig6 = sim.plot_gnc_data_multi([
    'navigation.actual_states[15]',
    'navigation.actual_states[16]',
    'navigation.actual_states[17]'
])

fig7 = sim.plot_gnc_data_multi([
    'diagnosis.diagnosis.diagnosis_theta[0]',
    'diagnosis.diagnosis.diagnosis_theta[1]',
    'diagnosis.diagnosis.diagnosis_theta[3]',
    'diagnosis.diagnosis.diagnosis_theta[4]',
])

fig7.axes[0].set_ylim(0, 1.1)

fig8 = sim.plot_gnc_data_multi([
    'diagnosis.diagnosis.diagnosis_theta_cov[0]',
    'diagnosis.diagnosis.diagnosis_theta_cov[1]',
    'diagnosis.diagnosis.diagnosis_theta_cov[3]',
    'diagnosis.diagnosis.diagnosis_theta_cov[4]',
])
# fig8 = sim.plot_gnc_data_multi([
#     'diagnosis.diagnosis.residuals[0]',
#     'diagnosis.diagnosis.residuals[1]',
#     'diagnosis.diagnosis.residuals[2]',
#     'diagnosis.diagnosis.residuals[3]',
#     'diagnosis.diagnosis.residuals[4]',
#     'diagnosis.diagnosis.residuals[5]',
#     'diagnosis.diagnosis.residuals[6]',
#     'diagnosis.diagnosis.residuals[7]',
# ])
# fig9 = sim.plot_gnc_data_multi([
#     'diagnosis.diagnosis.prediction_error[0]',
#     'diagnosis.diagnosis.prediction_error[1]',
#     'diagnosis.diagnosis.prediction_error[2]',
#     'diagnosis.diagnosis.prediction_error[3]',
#     'diagnosis.diagnosis.prediction_error[4]',
#     'diagnosis.diagnosis.prediction_error[5]',
#     'diagnosis.diagnosis.prediction_error[6]',
#     'diagnosis.diagnosis.prediction_error[7]',
# ])

plt.show(block=True)