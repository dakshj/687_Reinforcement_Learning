import itertools
import math

import matplotlib.pyplot as plt
import numpy as np

from agent import cartpole
from agent import gridworld

ALPHA_START = -15
ALPHA_END = -1
ALPHA_VALUES = np.arange(ALPHA_START, ALPHA_END + 1)
WEIGHT_UPDATE_NUM_EPISODES = 100
MSE_CALC_NUM_EPISODES = 100
FOURIER_BASIS_VALUES = [3, 5]
ENV_VALUES = [cartpole.ENV, gridworld.ENV]
MAX_TIME_STEPS = 1010
MAX_ACCEPTABLE_MSE = 10000


def execute(env: str, alpha: float, agent_execute_func, fourier_basis_n: int = None,
            weight_update_episodes: int = WEIGHT_UPDATE_NUM_EPISODES,
            mse_calc_episodes: int = MSE_CALC_NUM_EPISODES) -> float:
    """

    :return: Mean-squared TD error over all episodes and all time steps per episode
    """
    fourier_arr = None
    weights = None
    v_arr = None
    v_prev = None
    state_prev = None

    # Save TD Errors for each episode X each time step
    td_errs = np.zeros((mse_calc_episodes, MAX_TIME_STEPS * 100))

    for time_step, episode, state, reward, gamma in agent_execute_func(
            weight_update_episodes + mse_calc_episodes):

        v = None
        phi = None

        if env == cartpole.ENV:
            if fourier_arr is None:
                fourier_arr = get_nth_order_fourier_basis(state, fourier_basis_n)

            phi = np.cos(math.pi * np.dot(fourier_arr, state))

            if weights is None:
                weights = get_weights_zeros(np.shape(phi)[0])

            v = np.dot(weights, phi)

        elif env == gridworld.ENV:
            if v_arr is None:
                v_arr = np.zeros(gridworld.GRID_SHAPE)

        if time_step != 0:
            td_err = None

            if env == cartpole.ENV:
                td_err = reward + (gamma * v) - v_prev
            elif env == gridworld.ENV:
                td_err = reward + (gamma * v_arr[state]) - v_arr[state_prev]

            if 0 <= episode < weight_update_episodes:
                if env == cartpole.ENV:
                    weights += alpha * td_err * phi
                elif env == gridworld.ENV:
                    v_arr[state_prev] += alpha * td_err

            elif weight_update_episodes <= episode < \
                    (weight_update_episodes + mse_calc_episodes):
                td_errs[episode - weight_update_episodes, time_step] = td_err

        v_prev = v
        state_prev = state

    # Calculate and return mean-squared TD error
    return np.average(np.sum(td_errs ** 2, axis=1))


def get_nth_order_fourier_basis(policy: np.ndarray, fourier_basis_n: int) -> np.ndarray:
    return np.array(list(itertools.product(range(fourier_basis_n + 1),
        repeat=policy.shape[0])))


def get_weights_zeros(n: int):
    return np.zeros((n,))


def plot(data):
    labels = ['CartPole Fourier Basis 3', 'CartPole Fourier Basis 5', 'GridWorld']
    for row, label in zip(data, labels):
        x, y = map(list, zip(*row))
        plt.plot([round(x_ith) for x_ith in x], y, label=label)
        plt.xticks(np.arange(min(x), max(x) + 1))

    plt.legend(loc='upper left')

    plt.xlabel('Alpha (in log10 space)')
    plt.ylabel('Mean-Squared TD Error')

    plt.show()


def get_agent_execute_func(env: str):
    if env == cartpole.ENV:
        return cartpole.execute
    elif env == gridworld.ENV:
        return gridworld.execute


def execute_all():
    results = []

    for env in ENV_VALUES:
        for fourier in FOURIER_BASIS_VALUES:

            curr_results = []
            for alpha in ALPHA_VALUES:
                mse = execute(env, math.pow(10, alpha), fourier_basis_n=fourier,
                    agent_execute_func=get_agent_execute_func(env))

                if mse <= MAX_ACCEPTABLE_MSE:
                    curr_results.append((alpha, mse))

                # End for alpha

            results.append(curr_results)

            if env == gridworld.ENV:
                # Execute only once in the fourier loop for gridworld
                # (since it is unrequired for gridworld)
                break

    return results


if __name__ == '__main__':
    plot(execute_all())
