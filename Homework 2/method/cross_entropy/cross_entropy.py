import time

import numpy as np

from agent import cartpole
from agent import gridworld
from agent.cartpole import generate_initial_cartpole_policy
from agent.gridworld import generate_initial_gridworld_tabular_softmax_policy, \
    convert_theta_to_table

EPSILON = 0.0001


def cross_entropy(while_limit, K, K_e, N, trial, trials_total, env: str):
    if env == gridworld.ENV:
        theta = generate_initial_gridworld_tabular_softmax_policy()
        sigma = np.identity(92)

    elif env == cartpole.ENV:
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

            if env == gridworld.ENV:
                table = convert_theta_to_table(theta_k)
                episodes_results = gridworld.execute(episodes=N, policy_table=table)

            elif env == cartpole.ENV:
                episodes_results = cartpole.execute(episodes=N, policy=theta_k)

            trial_results.extend(episodes_results)

            list_of__theta_k__vs__J_k_hat.append((theta_k,

                                                  # J_k_hat
                                                  np.mean(episodes_results)
                                                  ))

            # End K loop

        list_of__theta_k__vs__J_k_hat.sort(key=lambda x: x[1], reverse=True)

        top_theta_k = [x for (x, _) in list_of__theta_k__vs__J_k_hat][:K_e]

        theta_k_sum = np.sum(top_theta_k, axis=0)

        theta = 1 / K_e * theta_k_sum

        diff = top_theta_k - theta
        summation_part = np.dot(diff.T, diff)

        if env == gridworld.ENV:
            identity = np.identity(92)
        elif env == cartpole.ENV:
            identity = np.identity(4)

        # noinspection PyUnboundLocalVariable
        sigma = (1 / (EPSILON + K_e)) * ((EPSILON * identity) + summation_part)

        # End While loop

    return trial_results
