import time

import numpy as np

from agent import cartpole
from agent import gridworld
from agent.cartpole import generate_initial_cartpole_policy
from agent.gridworld import generate_initial_gridworld_tabular_softmax_policy
from util.softmax import softmax


def convert_theta_to_table(theta):
    return softmax(X=np.reshape(theta, (-1, 4)), theta=0.5, axis=1)


# Environments
ENV_GRIDWORLD, ENV_CARTPOLE = 'gridworld', 'cartpole'

# Common constants
EPSILON = 0.0001


def cross_entropy(while_limit, K, K_e, N, trial, trials_total, env: str):
    if env == ENV_GRIDWORLD:
        theta = generate_initial_gridworld_tabular_softmax_policy()
        sigma = np.identity(92)

    elif env == ENV_CARTPOLE:
        theta = generate_initial_cartpole_policy()
        sigma = np.identity(4)

    list_of__theta_k__vs__J_k_hat = []

    trial_results = []

    exec_time = time.time()

    for while_i in range(while_limit):
        print('{} / {} in trial {} / {} (time = {} s)'
            .format(while_i + 1, while_limit, trial + 1, trials_total,
            round(time.time() - exec_time, 2)))
        exec_time = time.time()

        for _ in range(K):
            # noinspection PyUnboundLocalVariable
            theta_k = np.random.multivariate_normal(theta, sigma)

            if env == ENV_GRIDWORLD:
                table = convert_theta_to_table(theta_k)
                episodes_results = gridworld.execute(episodes=N, policy_table=table)

            elif env == ENV_CARTPOLE:
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

        top_policy_samples = filtered_theta_k_list - theta
        summation_part = np.dot(top_policy_samples.T, top_policy_samples)

        if env == ENV_GRIDWORLD:
            identity = np.identity(92)
        elif env == ENV_CARTPOLE:
            identity = np.identity(4)

        # noinspection PyUnboundLocalVariable
        sigma = (1 / (EPSILON + K_e)) * ((EPSILON * identity) + summation_part)

        # End While loop

    return trial_results
