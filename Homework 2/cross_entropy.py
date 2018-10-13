import os
import time

import numpy as np

import cartpole
import gridworld


def softmax(X, theta=1.0, axis=None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)

    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)

    # finally: divide element-wise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p


def convert_theta_to_table(theta):
    return softmax(X=np.reshape(theta, (-1, 4)), theta=0.5, axis=1)


def generate_initial_gridworld_tabular_softmax_policy():
    return np.random.uniform(0, 1, (92,))


def generate_initial_cartpole_policy():
    return np.random.uniform(-10, 10, (4,))


# Environments
GRIDWORLD, CARTPOLE = 'gridworld', 'cartpole'

# Common constants
EPSILON = 0.0001

# Gridworld Constants
GRID_TRIALS_DIR = '{}_cross_entropy_trials'.format(GRIDWORLD)
GRID_TRIALS = 20
GRID_WHILE_LOOP_ITERATIONS_VALUES = [100]
GRID_K_VALUES = [300]
GRID_K_e_VALUES = [30]
GRID_N_VALUES = [50]

# Cartpole Constants
CART_GRID_TRIALS_DIR = '{}_cross_entropy_trials'.format(CARTPOLE)
CART_TRIALS = 20
CART_WHILE_LOOP_ITERATIONS_VALUES = [100]
CART_K_VALUES = [300]
CART_K_e_VALUES = [30]
CART_N_VALUES = [50]


def save_trial(np_arr):
    if not os.path.exists(GRID_TRIALS_DIR):
        os.mkdir(GRID_TRIALS_DIR)

    np.save('{}/trial_{}'.format(GRID_TRIALS_DIR, time.time()), np_arr)


def execute_gridworld(env: str):
    for trial in range(GRID_TRIALS):
        for while_hyp in GRID_WHILE_LOOP_ITERATIONS_VALUES:
            for K_hyp in GRID_K_VALUES:
                for K_e_hyp in GRID_K_e_VALUES:
                    for N_hyp in GRID_N_VALUES:
                        save_trial(
                            cross_entropy(while_hyp, K_hyp, K_e_hyp, N_hyp, trial, env))


def execute_cartpole(env: str):
    for trial in range(CART_TRIALS):
        for while_hyp in CART_WHILE_LOOP_ITERATIONS_VALUES:
            for K_hyp in CART_K_VALUES:
                for K_e_hyp in CART_K_e_VALUES:
                    for N_hyp in CART_N_VALUES:
                        save_trial(
                            cross_entropy(while_hyp, K_hyp, K_e_hyp, N_hyp, trial, env))


def execute(env: str):
    if env == GRIDWORLD:
        execute_gridworld(env)
    elif env == CARTPOLE:
        execute_cartpole(env)


def cross_entropy(while_limit, K, K_e, N, trial, env: str):
    if env == GRIDWORLD:
        theta = generate_initial_gridworld_tabular_softmax_policy()
        sigma = np.identity(92)

    elif env == CARTPOLE:
        theta = generate_initial_cartpole_policy()
        sigma = np.identity(4)

    list_of__theta_k__vs__J_k_hat = []

    trial_results = []

    exec_time = time.time()

    for while_i in range(while_limit):
        print('{} / {} in trial {} / {} (time = {} s)'
            .format(while_i, while_limit, trial, GRID_TRIALS, round(time.time() - exec_time, 2)))
        exec_time = time.time()

        for _ in range(K):
            # noinspection PyUnboundLocalVariable
            theta_k = np.random.multivariate_normal(theta, sigma)

            table = convert_theta_to_table(theta_k)

            if env == GRIDWORLD:
                episodes_results = gridworld.execute(episodes=N, policy_table=table)

            elif env == CARTPOLE:
                episodes_results = cartpole.execute(episodes=N, policy=theta_k)

            trial_results.extend(episodes_results)

            list_of__theta_k__vs__J_k_hat.append((theta_k,

                                                  # J_k_hat
                                                  np.mean(episodes_results)
                                                  ))

            # End K loop

        list_of__theta_k__vs__J_k_hat.sort(key=lambda x: x[1], reverse=True)

        filtered_theta_k_list = [x for (x, _) in list_of__theta_k__vs__J_k_hat][:K_e]

        theta_k_sum = np.sum(filtered_theta_k_list, axis=0)

        theta = 1 / K_e * theta_k_sum

        summation_part = np.sum(
            [(x - theta) * np.transpose((x - theta)) for x in filtered_theta_k_list],
            axis=0
        )

        sigma = (1 / (EPSILON + K_e)) * ((EPSILON * np.identity(92)) + summation_part)

        # End While loop

    return trial_results


if __name__ == '__main__':
    execute(CARTPOLE)
