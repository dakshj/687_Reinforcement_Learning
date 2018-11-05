import itertools
import math

import matplotlib.pyplot as plt
import numpy as np

from agent import cartpole
from agent import gridworld

ALPHA_START = -10
ALPHA_END = -1
ALPHA_VALUES = [math.pow(10, x) for x in np.arange(ALPHA_START, ALPHA_END + 1)]
WEIGHT_UPDATE_NUM_EPISODES = 100
MSE_CALC_NUM_EPISODES = 100
FOURIER_BASIS_VALUES = [3, 5]
ENV_VALUES = [cartpole.ENV, gridworld.ENV]


def execute(env: str, alpha: float, fourier_basis_n: int = None,
            num_episodes_for_weight_update: int = WEIGHT_UPDATE_NUM_EPISODES,
            num_episodes_for_mse_calc: int = MSE_CALC_NUM_EPISODES) -> float:
    policy = get_policy(env)

    fourier_arr = get_nth_order_fourier_basis(policy, fourier_basis_n)
    phi = [math.cos(math.pi * np.dot(c_i, policy)) for c_i in fourier_arr]
    weights = get_random_weights(np.shape(phi)[0])

    for _ in range(num_episodes_for_weight_update):
        # TODO
        td_err = None
        weights += alpha * td_err * None

    mse = 0.

    for _ in range(num_episodes_for_mse_calc):
        # TODO Calculate td_err
        td_err = 0.

        mse += td_err

    return mse / num_episodes_for_mse_calc


def get_policy(env: str):
    if env == gridworld.ENV:
        return gridworld.generate_random_gridworld_tabular_softmax_policy()

    if env == cartpole.ENV:
        return cartpole.generate_random_cartpole_policy()


def get_nth_order_fourier_basis(policy: np.ndarray, fourier_basis_n: int) -> list:
    return list(itertools.product(range(fourier_basis_n + 1), repeat=policy.shape[0]))


def get_random_weights(n: int):
    return np.random.uniform(-10, 10, (n,))


def plot(data):
    labels = ['CartPole Fourier Basis 3', 'CartPole Fourier Basis 5', 'GridWorld']
    for row, label in zip(data, labels):
        x, y = map(list, zip(*row))
        print(type(x))
        print(x)
        plt.plot(x, y, label=label)
        plt.xticks(np.arange(min(x), max(x) + 1))

    plt.legend(loc='upper left')

    plt.xlabel('Alpha')
    plt.ylabel('Mean-Squared TD Error')

    plt.show()


def execute_all():
    results = []

    for env in ENV_VALUES:
        for fourier in FOURIER_BASIS_VALUES:

            curr_results = []
            for alpha in ALPHA_VALUES:
                mse = execute(env, alpha, fourier_basis_n=fourier)
                curr_results.append((math.log(alpha, 10), mse))

            results.append(curr_results)

            if env == gridworld.ENV:
                # Execute only once in the fourier loop for gridworld
                # (since it is unrequired for gridworld)
                break

    return results


if __name__ == '__main__':
    plot(execute_all())
