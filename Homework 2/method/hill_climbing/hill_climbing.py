import time

import numpy as np

from agent import cartpole
from agent import gridworld
from agent.cartpole import generate_initial_cartpole_policy
from agent.gridworld import convert_theta_to_table
from agent.gridworld import generate_initial_gridworld_tabular_softmax_policy

EPSILON = 0.0001


def get_returns(env: str, episodes: int, policy: np.ndarray):
    """

    :rtype: Tuple of {J_hat, array-of-rewards}
    """
    if env == gridworld.ENV:
        table = convert_theta_to_table(policy)
        results = gridworld.execute(episodes=episodes, policy_table=table)

    elif env == cartpole.ENV:
        results = cartpole.execute(episodes=episodes, policy=policy)

    # noinspection PyUnboundLocalVariable
    return results, np.mean(results)


def hill_climbing(while_limit: int, sigma: float, N: int, trial, trials_total, env: str):
    if env == gridworld.ENV:
        theta = generate_initial_gridworld_tabular_softmax_policy()
        sigma = sigma * np.identity(92)

    elif env == cartpole.ENV:
        theta = generate_initial_cartpole_policy()
        sigma = sigma * np.identity(4)

    trial_results = []

    # noinspection PyUnboundLocalVariable
    episodes_results, J_hat = get_returns(env=env, episodes=N, policy=theta)
    trial_results.extend(episodes_results)

    exec_time = time.time()

    for while_i in range(while_limit):
        print('{} / {} in trial {} / {} (time = {} s)'
            .format(while_i + 1, while_limit, trial + 1, trials_total,
            round(time.time() - exec_time, 2)))
        exec_time = time.time()

        theta_prime = np.random.multivariate_normal(theta, sigma)

        episodes_results, J_hat_prime = get_returns(env=env, episodes=N, policy=theta_prime)
        trial_results.extend(episodes_results)

        if J_hat_prime > J_hat:
            theta = theta_prime
            J_hat = J_hat_prime

        # End While loop
        pass

    return trial_results
