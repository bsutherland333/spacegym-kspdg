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
from matplotlib import pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

# instantiate and reset the environment to populate game
env = PE1_E1_I3_Env(episode_timeout=600.0, capture_dist=5.0)
obs, info = env.reset()

# State constants
mu = C.KERBIN.MU  # m^3/s^2
time_final = 300.0  # seconds

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

def calculate_K(obs):
    def A(t):
        initial_state = [obs[i] for i in range(3, 15)]

        def dynamics(t, x):
            x_p = x[0]
            y_p = x[1]
            z_p = x[2]
            x_e = x[6]
            y_e = x[7]
            z_e = x[8]

            dx_p_dt = x[3]
            dy_p_dt = x[4]
            dz_p_dt = x[5]
            dx_e_dt = x[9]
            dy_e_dt = x[10]
            dz_e_dt = x[11]

            dvx_p_dt = mu*x_p / (x_p**2 + y_p**2 + z_p**2)**(3/2)
            dvy_p_dt = mu*y_p / (x_p**2 + y_p**2 + z_p**2)**(3/2)
            dvz_p_dt = mu*z_p / (x_p**2 + y_p**2 + z_p**2)**(3/2)
            dvx_e_dt = mu*x_e / (x_e**2 + y_e**2 + z_e**2)**(3/2)
            dvy_e_dt = mu*y_e / (x_e**2 + y_e**2 + z_e**2)**(3/2)
            dvz_e_dt = mu*z_e / (x_e**2 + y_e**2 + z_e**2)**(3/2)

            return [dx_p_dt, dy_p_dt, dz_p_dt, dvx_p_dt, dvy_p_dt, dvz_p_dt,
                    dx_e_dt, dy_e_dt, dz_e_dt, dvx_e_dt, dvy_e_dt, dvz_e_dt]

        # Integrate the dynamics
        t_span = [0, t]
        sol = solve_ivp(dynamics, t_span, initial_state)
        current_state = sol.y[:, -1]

        x_p_curr = current_state[0]
        y_p_curr = current_state[1]
        z_p_curr = current_state[2]
        x_e_curr = current_state[6]
        y_e_curr = current_state[7]
        z_e_curr = current_state[8]

        # Calculate the partial derivatives of the dynamics
        partial_f_x = mu*(-2*x_p_curr**2 + y_p_curr**2 + z_p_curr**2) / (x_p_curr**2 + y_p_curr**2 + z_p_curr**2)**(5/2) - \
            mu*(-2*x_e_curr**2 + y_e_curr**2 + z_e_curr**2) / (x_e_curr**2 + y_e_curr**2 + z_e_curr**2)**(5/2)
        partial_f_y = mu*(x_p_curr**2 - 2*y_p_curr**2 + z_p_curr**2) / (x_p_curr**2 + y_p_curr**2 + z_p_curr**2)**(5/2) - \
            mu*(x_e_curr**2 - 2*y_e_curr**2 + z_e_curr**2) / (x_e_curr**2 + y_e_curr**2 + z_e_curr**2)**(5/2)
        partial_f_z = mu*(x_p_curr**2 + y_p_curr**2 - 2*z_p_curr**2) / (x_p_curr**2 + y_p_curr**2 + z_p_curr**2)**(5/2) - \
            mu*(x_e_curr**2 + y_e_curr**2 - 2*z_e_curr**2) / (x_e_curr**2 + y_e_curr**2 + z_e_curr**2)**(5/2)

        # State space model
        A = np.array([[0, 1, 0, 0, 0, 0],
                      [partial_f_x, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0],
                      [0, 0, partial_f_y, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, partial_f_z, 0]])
        return A

    mass = float(obs[1])
    B = np.array([[0, 0, 0],
                  [1 / mass, 0, 0],
                  [0, 0, 0],
                  [0, 1 / mass, 0],
                  [0, 0, 0],
                  [0, 0, 1 / mass]])

    def dre(t, P_flat):
        P = P_flat.reshape(6, 6)
        dP = -(A(t).T @ P + P @ A(t) - P @ B @ np.linalg.inv(R) @ B.T @ P + Q)
        return dP.flatten()

    # Integrate backwards from tf to now
    time_curr = obs[0]
    P_tf = np.zeros((6, 6)).flatten()
    sol = solve_ivp(dre, [time_final, time_curr], P_tf)

    # Compute LQR gain
    P = sol.y[:, -1].reshape(6, 6)
    K = np.linalg.inv(R) @ B.T @ P

    return K

is_done = False
time_hist = []
propellant_mass_hist = []
rel_pos_hist = []
rel_vel_hist = []
u_hist = []
u_saturated_hist = []

time_at_k_update = 0.0
K = calculate_K(obs)
try:
    while not is_done:
        # get observations
        time_curr = obs[0]
        propellant_mass = obs[2]
        pos_prime = np.array(obs[3:6]) - np.array(obs[9:12])
        vel_prime = np.array(obs[6:9]) - np.array(obs[12:15])
        state = np.hstack((pos_prime.reshape(-1, 1), vel_prime.reshape(-1, 1))).reshape(-1, 1)

        # store observations
        time_hist.append(time_curr)
        propellant_mass_hist.append(propellant_mass)
        rel_pos_hist.append(pos_prime)
        rel_vel_hist.append(vel_prime)

        # calculate LQR gain
        if (time_curr - time_at_k_update) > 5.0:
            time_at_k_update = time_curr
            K = calculate_K(obs)

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
