# Copyright (c) 2024, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).
# SPDX-License-Identifier: MIT

"""
Instructions to Run:
- Start KSP game application.
- Select Start Game > Play Missions > Community Created > pe1_i3 > Continue
- In kRPC dialog box click Add server. Select Show advanced settings and select Auto-accept new clients. Then select Start Server
- In a terminal, run this script

"""

from kspdg.pe1.e1_envs import PE1_E1_I3_Env
import kspdg.utils.constants as C
import numpy as np
import control as ctrl
from matplotlib import pyplot as plt

# instantiate and reset the environment to populate game
env = PE1_E1_I3_Env(episode_timeout=600.0, capture_dist=5.0)
obs, info = env.reset()

# State constants
mu = C.KERBIN.MU  # m^3/s^2
mass = float(obs[1])  # kg

x_p = obs[3]
x_e = obs[9]
y_p = obs[4]
y_e = obs[10]
z_p = obs[5]
z_e = obs[11]

# Calculate the partial derivatives of the dynamics
partial_f_x = mu*(-2*x_p**2 + y_p**2 + z_p**2) / (x_p**2 + y_p**2 + z_p**2)**(5/2) - \
    mu*(-2*x_e**2 + y_e**2 + z_e**2) / (x_e**2 + y_e**2 + z_e**2)**(5/2)
partial_f_y = mu*(x_p**2 - 2*y_p**2 + z_p**2) / (x_p**2 + y_p**2 + z_p**2)**(5/2) - \
    mu*(x_e**2 - 2*y_e**2 + z_e**2) / (x_e**2 + y_e**2 + z_e**2)**(5/2)
partial_f_z = mu*(x_p**2 + y_p**2 - 2*z_p**2) / (x_p**2 + y_p**2 + z_p**2)**(5/2) - \
    mu*(x_e**2 + y_e**2 - 2*z_e**2) / (x_e**2 + y_e**2 + z_e**2)**(5/2)

# State space model
A = np.array([[0, 1, 0, 0, 0, 0],
              [partial_f_x, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 0],
              [0, 0, partial_f_y, 0, 0, 0],
              [0, 0, 0, 0, 0, 1],
              [0, 0, 0, 0, partial_f_z, 0]])
B = np.array([[0, 0, 0],
              [1 / mass, 0, 0],
              [0, 0, 0],
              [0, 1 / mass, 0],
              [0, 0, 0],
              [0, 0, 1 / mass]])
C = np.eye(6)
D = np.array([[0, 0, 0]])

# LQR parameters
pos_weight = 2.5
vel_weight = 50
u_weight = 0.5

Q = np.array([[pos_weight, 0, 0, 0, 0, 0],
              [0, vel_weight, 0, 0, 0, 0],
              [0, 0, pos_weight, 0, 0, 0],
              [0, 0, 0, vel_weight, 0, 0],
              [0, 0, 0, 0, pos_weight, 0],
              [0, 0, 0, 0, 0, vel_weight]])
R = np.eye(3) * u_weight

# Compute LQR gain
K, S, E = ctrl.lqr(A, B, Q, R)

is_done = False
time_hist = []
propellant_mass_hist = []
rel_pos_hist = []
rel_vel_hist = []
u_hist = []
u_saturated_hist = []
try:
    while not is_done:
        # get observations
        time = obs[0]
        propellant_mass = obs[2]
        pos_prime = np.array(obs[3:6]) - np.array(obs[9:12])
        vel_prime = np.array(obs[6:9]) - np.array(obs[12:15])
        state = np.hstack((pos_prime.reshape(-1, 1), vel_prime.reshape(-1, 1))).reshape(-1, 1)

        # store observations
        time_hist.append(time)
        propellant_mass_hist.append(propellant_mass)
        rel_pos_hist.append(pos_prime)
        rel_vel_hist.append(vel_prime)

        # calculate control
        u = (-K @ state).flatten()
        u_saturated = np.clip(u, -env.agent_max_thrust_up, env.agent_max_thrust_up)
        u_hist.append(u)
        u_saturated_hist.append(u_saturated)

        # apply control
        env.logger.info(f"rel_pos: {pos_prime}, rel_vel: {vel_prime}, u_saturated: {u_saturated}")
        act = {
            "burn_vec": [u_saturated.item(0), u_saturated.item(1), u_saturated.item(2), 0.1],
            "vec_type": 1,
            "ref_frame": 1
        }
        obs, rew, is_done, info = env.step(act)
except KeyboardInterrupt:
    pass

# close the environments to cleanup any processes
env.close()

### plot the results ###

# convert lists to numpy arrays
rel_pos_array = np.array(rel_pos_hist)
rel_vel_array = np.array(rel_vel_hist)
u_array = np.array(u_hist)
u_saturated_array = np.array(u_saturated_hist)

# create plot object
fig, axs = plt.subplots(3, 4, figsize=(16, 12))

# set formatting
axs[0, 0].set_title('x (s)')
axs[0, 1].set_title('y (s)')
axs[0, 2].set_title('z (s)')
axs[0, 3].set_title('magnitude (s)')
axs[0, 0].set_ylabel('relative position (m)')
axs[1, 0].set_ylabel('relative velocity (m/s)')
axs[2, 0].set_ylabel('control (N)')
for i in range(axs.shape[0]):
    for j in range(axs.shape[1]):
        axs[i, j].grid()

# plot relative position
axs[0, 0].plot(time_hist, rel_pos_array[:, 0], color='b')
axs[0, 1].plot(time_hist, rel_pos_array[:, 1], color='b')
axs[0, 2].plot(time_hist, rel_pos_array[:, 2], color='b')
axs[0, 3].plot(time_hist, np.linalg.norm(rel_pos_array, axis=1), color='b')

# plot relative velocity
axs[1, 0].plot(time_hist, rel_vel_array[:, 0], color='b')
axs[1, 1].plot(time_hist, rel_vel_array[:, 1], color='b')
axs[1, 2].plot(time_hist, rel_vel_array[:, 2], color='b')
axs[1, 3].plot(time_hist, np.linalg.norm(rel_vel_array, axis=1), color='b')

# plot control
axs[2, 0].plot(time_hist, u_array[:, 0], color='b', label='u')
axs[2, 0].plot(time_hist, u_saturated_array[:, 0], color='r', label='u saturated')
axs[2, 0].legend()
axs[2, 1].plot(time_hist, u_array[:, 1], color='b')
axs[2, 1].plot(time_hist, u_saturated_array[:, 1], color='r')
axs[2, 2].plot(time_hist, u_array[:, 2], color='b')
axs[2, 2].plot(time_hist, u_saturated_array[:, 2], color='r')
axs[2, 3].plot(time_hist, np.linalg.norm(u_array, axis=1), color='b')
axs[2, 3].plot(time_hist, np.linalg.norm(u_saturated_array, axis=1), color='r')

plt.tight_layout()
plt.show()
