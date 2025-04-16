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

import control as ctrl
from kspdg.pe1.e1_envs import PE1_E1_I3_Env
from kspdg.pe1.pe1_base import PursuitEvadeGroup1Env
import kspdg.utils.constants as C
from matplotlib import pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

# instantiate and reset the environment to populate game
MU = C.KERBIN.MU  # m^3/s^2
MAX_THRUST = np.abs(PursuitEvadeGroup1Env.PARAMS.PURSUER.RCS.VACUUM_MAX_THRUST_UP)
MAX_FUEL_CONSUMPION = PursuitEvadeGroup1Env.PARAMS.PURSUER.RCS.VACUUM_MAX_FUEL_CONSUMPTION_UP
max_time = 180.0  # seconds
env = PE1_E1_I3_Env(episode_timeout=max_time + 60, capture_dist=5.0)
obs, info = env.reset()
mass = float(obs[1])

def dynamics(t, x, thrust):
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

    dvx_p_dt = -MU*x_p / (x_p**2 + y_p**2 + z_p**2)**(3/2) + thrust[0] / mass
    dvy_p_dt = -MU*y_p / (x_p**2 + y_p**2 + z_p**2)**(3/2) + thrust[1] / mass
    dvz_p_dt = -MU*z_p / (x_p**2 + y_p**2 + z_p**2)**(3/2) + thrust[2] / mass
    dvx_e_dt = -MU*x_e / (x_e**2 + y_e**2 + z_e**2)**(3/2)
    dvy_e_dt = -MU*y_e / (x_e**2 + y_e**2 + z_e**2)**(3/2)
    dvz_e_dt = -MU*z_e / (x_e**2 + y_e**2 + z_e**2)**(3/2)

    return np.array([dx_p_dt, dy_p_dt, dz_p_dt, dvx_p_dt, dvy_p_dt, dvz_p_dt,
                     dx_e_dt, dy_e_dt, dz_e_dt, dvx_e_dt, dvy_e_dt, dvz_e_dt])

def cost_function(input, initial_state):
    # Input is assumed to be in minutes and normalized u, with size 9

    desired_first_step = 2.5
    final_time = (input[4] + input[8]) * 60

    # Initial burn
    t_span = [0, input[3] * 60]
    thrust = input[0:3] * MAX_THRUST
    first_step = np.min([desired_first_step, input[3] * 60 - 0.1])
    first_step = first_step if first_step > 0 else None
    pre_drift_state = solve_ivp(dynamics, t_span, initial_state, first_step=first_step, args=(thrust,)).y[:, -1]

    # Drift
    t_span = [input[3] * 60, input[4] * 60]
    thrust = np.zeros(3)
    first_step = np.min([desired_first_step, (input[4] - input[3]) * 60 - 0.1])
    first_step = first_step if first_step > 0 else None
    post_drift_state = solve_ivp(dynamics, t_span, pre_drift_state, first_step=first_step, args=(thrust,)).y[:, -1]

    # Final burn
    t_span = [input[4] * 60, final_time]
    first_step = np.min([desired_first_step, input[8] * 60 - 0.1])
    first_step = first_step if first_step > 0 else None
    final_state = solve_ivp(dynamics, t_span, post_drift_state, first_step=first_step, args=(input,)).y[:, -1]

    # Calculate the cost
    rel_pos = np.array(final_state[0:3]) - np.array(final_state[6:9])
    rel_vel = np.array(final_state[3:6]) - np.array(final_state[9:12])

    initial_fuel_consumpion = np.abs(input[0:3]).sum() * input[3] * 60 * MAX_FUEL_CONSUMPION
    final_fuel_consumpion = np.abs(input[5:8]).sum() * input[8] * 60 * MAX_FUEL_CONSUMPION
    total_fuel_consumpion = initial_fuel_consumpion + final_fuel_consumpion

    return (0.1 * np.linalg.norm(rel_pos))**2.0 + (0.5 * np.linalg.norm(rel_vel))**1.5 + \
        (0.1 * total_fuel_consumpion)**1.25 + (0.01 * final_time)

def get_trajectory(initial_time, initial_state, input):
    # Input is assumed to be in seconds and newtons and size 10

    # Pre-manuever drift
    t_span = [initial_time, input[0]]
    t_eval = np.linspace(initial_time, input[0], int(input[0] * 2))
    thrust = np.zeros(3)
    pre_manuever_trajectory_sol = solve_ivp(dynamics, t_span, initial_state, t_eval=t_eval, args=(thrust,))
    if len(pre_manuever_trajectory_sol.y) == 0:
        pre_manuever_trajectory_sol.y = np.zeros((12, 0))
        pre_manuever_state = initial_state
    else:
        pre_manuever_state = pre_manuever_trajectory_sol.y[:, -1]
        pre_manuever_trajectory_sol.y = pre_manuever_trajectory_sol.y[:, :-1]
        pre_manuever_trajectory_sol.t = pre_manuever_trajectory_sol.t[:-1]

    # Initial burn
    t_span = [input[0], input[0] + input[4]]
    t_eval = np.linspace(input[0], input[0] + input[4], int(input[4] * 2))
    thrust = input[1:4]
    initial_trajectory_sol = solve_ivp(dynamics, t_span, pre_manuever_state, t_eval=t_eval, args=(thrust,))
    if len(initial_trajectory_sol.y) == 0:
        initial_trajectory_sol.y = np.zeros((12, 0))
        pre_drift_state = pre_manuever_state
    else:
        pre_drift_state = initial_trajectory_sol.y[:, -1]
        initial_trajectory_sol.y = initial_trajectory_sol.y[:, :-1]
        initial_trajectory_sol.t = initial_trajectory_sol.t[:-1]

    # Drift
    t_span = [input[0] + input[4], input[5]]
    t_eval = np.linspace(input[0] + input[4], input[5], int((input[5] - input[4] - input[0]) * 2))
    thrust = np.zeros(3)
    drift_trajectory_sol = solve_ivp(dynamics, t_span, pre_drift_state, t_eval=t_eval, args=(thrust,))
    if len(drift_trajectory_sol.y) == 0:
        drift_trajectory_sol.y = np.zeros((12, 0))
        post_drift_state = pre_drift_state
    else:
        post_drift_state = drift_trajectory_sol.y[:, -1]
        drift_trajectory_sol.y = drift_trajectory_sol.y[:, :-1]
        drift_trajectory_sol.t = drift_trajectory_sol.t[:-1]

    # Final burn
    t_span = [input[5], input[5] + input[9]]
    t_eval = np.linspace(input[5], input[5] + input[9], int(input[9] * 2))
    thrust = input[1:4]
    final_trajectory_sol = solve_ivp(dynamics, t_span, post_drift_state, t_eval=t_eval, args=(thrust,))
    if len(final_trajectory_sol.y) == 0:
        final_trajectory_sol.y = np.zeros((12, 0))
        final_state = post_drift_state
    else:
        final_state = final_trajectory_sol.y[:, -1]
        final_trajectory_sol.y = final_trajectory_sol.y[:, :-1]
        final_trajectory_sol.t = final_trajectory_sol.t[:-1]

    # Post burn drift
    t_span = [input[5] + input[9], input[5] + input[9] + 60]
    t_eval = np.linspace(input[5] + input[9], input[5] + input[9] + 60, int(60 * 2))
    thrust = np.zeros(3)
    post_burn_trajectory_sol = solve_ivp(dynamics, t_span, final_state, t_eval=t_eval, args=(thrust,))
    if len(post_burn_trajectory_sol.y) == 0:
        post_burn_trajectory_sol.y = np.zeros((12, 0))

    full_trajectory = np.hstack((pre_manuever_trajectory_sol.y, initial_trajectory_sol.y,
                                 drift_trajectory_sol.y, final_trajectory_sol.y, post_burn_trajectory_sol.y))
    full_time = np.hstack((pre_manuever_trajectory_sol.t, initial_trajectory_sol.t,
                           drift_trajectory_sol.t, final_trajectory_sol.t, post_burn_trajectory_sol.t))

    return full_time, full_trajectory

def get_expected_state_command(t, trajectory, input):
    # Input is assumed to be in seconds and newtons, with initial start time as the first element
    # Trajectory is formatted as extended state from optimization
    # x is the state vector and u is the thrust vector as used by the pursuer control law

    t_array = trajectory[0]
    state_array = trajectory[1]

    # Get the current command based on the time
    if t >= input[5] + input[9]:
        thrust = np.zeros(3)
    elif t >= input[5]:
        thrust = input[6:9]
    elif t >= input[4] + input[0]:
        thrust = np.zeros(3)
    elif t >= input[0]:
        thrust = input[1:4]
    else:
        thrust = np.zeros(3)
    thrust = thrust.reshape(-1, 1)

    # Interpolate the current state in the trajectory
    idx = np.searchsorted(t_array, t)
    if (t - t_array[idx - 1]) < (t_array[idx] - t):
        idx_0 = idx - 2
        idx_1 = idx - 1
        idx_2 = idx
    else:
        idx_0 = idx - 1
        idx_1 = idx
        idx_2 = idx + 1

    l_0 = (t - t_array[idx_1]) * (t - t_array[idx_2]) / (t_array[idx_0] - t_array[idx_1]) / (t_array[idx_0] - t_array[idx_2])
    l_1 = (t - t_array[idx_0]) * (t - t_array[idx_2]) / (t_array[idx_1] - t_array[idx_0]) / (t_array[idx_1] - t_array[idx_2])
    l_2 = (t - t_array[idx_0]) * (t - t_array[idx_1]) / (t_array[idx_2] - t_array[idx_0]) / (t_array[idx_2] - t_array[idx_1])

    x = l_0 * state_array[:, idx_0] + l_1 * state_array[:, idx_1] + l_2 * state_array[:, idx_2]
    x = x[:6].reshape(-1, 1)

    return x, thrust

# Perform optimization
# input = start time, x thrust, y thrust, z thrust, duration (minutes, thrust -1 to 1)
# start time of initial burn is assumed to be zero, to reduce dimmentionality of
# optimization
optimal_input = np.array([0, 0, 0, 0.5, 2.0, 0, 0, 0, 0.5])
initial_time = obs[0]
initial_state = obs[3:15]
bounds = ((-1, 1), (-1, 1), (-1, 1), (0, None), (0, None), (-1, 1), (-1, 1), (-1, 1), (0, None))
constraints = ({'type': 'ineq', 'fun': lambda input: max_time - (input[4] + input[8]) * 60},
               {'type': 'ineq', 'fun': lambda input: input[4] - input[3]})
options = {'maxiter': 1000}
result = minimize(cost_function, optimal_input, bounds=bounds, constraints=constraints, options=options, args=(initial_state))
print("Optimization result:", result)
optimal_input = result.x

if not result.success or result.fun > 50:
    raise ValueError("Optimization failed")

# Convert input to seconds and newtons and add initial manuever start
optimal_input[0:3] = optimal_input[0:3] * MAX_THRUST
optimal_input[3:5] = optimal_input[3:5] * 60
optimal_input[5:8] = optimal_input[5:8] * MAX_THRUST
optimal_input[8] = optimal_input[8] * 60
obs = env.get_observation()
optimal_input = np.hstack((obs[0], optimal_input))
optimal_input[5] += obs[0]

# Get the initial and optimal trajectories
null_input = np.zeros(10)
null_input[0] = initial_time
null_input[4] = 30
null_input[5] = initial_time + 60
null_input[9] = optimal_input[5] + optimal_input[9] - initial_time - 60
initial_trajectory = get_trajectory(initial_time, initial_state, null_input)
optimal_trajectory = get_trajectory(initial_time, initial_state, optimal_input)

# State space model
A = np.array([[0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 1],
              [0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0]])
B = np.array([[0, 0, 0],
              [0, 0, 0],
              [0, 0, 0],
              [1 / mass, 0, 0],
              [0, 1 / mass, 0],
              [0, 0, 1 / mass]])
C = np.eye(6)
D = np.array([[0, 0, 0]])

# LQR parameters
pos_weight = 100
vel_weight = 10
u_weight = 1 / MAX_THRUST

Q = np.array([[pos_weight, 0, 0, 0, 0, 0],
              [0, pos_weight, 0, 0, 0, 0],
              [0, 0, pos_weight, 0, 0, 0],
              [0, 0, 0, vel_weight, 0, 0],
              [0, 0, 0, 0, vel_weight, 0],
              [0, 0, 0, 0, 0, vel_weight]])
R = np.eye(3) * u_weight

# Compute LQR gain
K, S, E = ctrl.lqr(A, B, Q, R)

is_done = False
time_hist = []
pos_hist = []
rel_pos_hist = []
rel_vel_hist = []
u_hist = []
u_saturated_hist = []
try:
    while not is_done:
        # get observations
        time = obs[0]
        rel_pos = np.array(obs[3:6]) - np.array(obs[9:12])
        rel_vel = np.array(obs[6:9]) - np.array(obs[12:15])
        x = np.array(obs[3:9]).reshape(-1, 1)

        # store observations
        time_hist.append(time)
        pos_hist.append(obs[3:6])
        rel_pos_hist.append(rel_pos)
        rel_vel_hist.append(rel_vel)

        # calculate control
        x_0, u_0 = get_expected_state_command(time, optimal_trajectory, optimal_input)
        u = (u_0 - K @ (x - x_0)).flatten()
        u_saturated = np.clip(u, -MAX_THRUST, MAX_THRUST)
        u_hist.append(u)
        u_saturated_hist.append(u_saturated)

        # apply control
        env.logger.info(f"state error: {(x - x_0).flatten()}")
        env.logger.info(f"u_s: {u_saturated.flatten()}, u_0: {u_0.flatten()}")
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
pos_array = np.array(pos_hist)
rel_pos_array = np.array(rel_pos_hist)
rel_vel_array = np.array(rel_vel_hist)
u_array = np.array(u_hist)
u_saturated_array = np.array(u_saturated_hist)

# Plot trajectory
fig, ax = plt.subplots(1, 3, figsize=(16, 8))
ax[0].plot(time_hist, pos_array[:, 0], color='b', label='Actual Trajectory')
ax[0].plot(initial_trajectory[0], initial_trajectory[1][0], color='r', label='Initial Trajectory')
ax[0].plot(optimal_trajectory[0], optimal_trajectory[1][0], color='g', label='Optimal Trajectory')
ax[0].set_title('x (s)')
ax[0].set_ylabel('Position (m)')
ax[0].grid()
ax[0].legend()
ax[1].plot(time_hist, pos_array[:, 1], color='b')
ax[1].plot(initial_trajectory[0], initial_trajectory[1][1], color='r')
ax[1].plot(optimal_trajectory[0], optimal_trajectory[1][1], color='g')
ax[1].set_title('y (s)')
ax[1].grid()
ax[2].plot(time_hist, pos_array[:, 2], color='b')
ax[2].plot(initial_trajectory[0], initial_trajectory[1][2], color='r')
ax[2].plot(optimal_trajectory[0], optimal_trajectory[1][2], color='g')
ax[2].set_title('z (s)')
ax[2].grid()
plt.suptitle('Trajectory of the spacecraft')
plt.tight_layout()
plt.show()

# Calculate error of actual and initial trajectory from optimal trajectory
actual_error = []
initial_error = []
for t, actual_pos in zip(time_hist, pos_array):
    actual_pos = np.array(actual_pos).reshape(-1, 1)
    optimal_pos = get_expected_state_command(t, optimal_trajectory, optimal_input)[0][:3]
    initial_pos = get_expected_state_command(t, initial_trajectory, np.zeros(10))[0][:3]
    actual_error.append(actual_pos - optimal_pos)
    initial_error.append(initial_pos - optimal_pos)
actual_error = np.array(actual_error).squeeze()
initial_error = np.array(initial_error).squeeze()

# Plot error
fig, ax = plt.subplots(1, 3, figsize=(16, 8))
ax[0].plot(time_hist, initial_error[:, 0], color='b', label='Initial Error')
ax[0].plot(time_hist, actual_error[:, 0], color='r', label='Actual Error')
ax[0].set_title('x (s)')
ax[0].set_ylabel('Position (m)')
ax[0].grid()
ax[0].legend()
ax[1].plot(time_hist, initial_error[:, 1], color='b')
ax[1].plot(time_hist, actual_error[:, 1], color='r')
ax[1].set_title('y (s)')
ax[1].grid()
ax[2].plot(time_hist, initial_error[:, 2], color='b')
ax[2].plot(time_hist, actual_error[:, 2], color='r')
ax[2].set_title('z (s)')
ax[2].grid()
plt.suptitle('Trajectory Trajectory Error')
plt.tight_layout()
plt.show()

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

plt.suptitle('LQR Control of the spacecraft')
plt.tight_layout()
plt.show()
