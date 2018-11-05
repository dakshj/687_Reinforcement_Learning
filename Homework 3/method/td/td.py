import itertools

import numpy as np

from agent import cartpole
from agent import gridworld


def td(env: str, alpha: float, num_episodes_weight_update: int,
       num_episodes_mse_calc: int, fourier_basis_n: int) -> float:
    policy = get_policy(env)
    fourier_arr = get_nth_order_fourier_basis(policy, fourier_basis_n)

    for _ in range(num_episodes_weight_update):
        pass

    mse = 0.

    for _ in range(num_episodes_mse_calc):
        # TODO Calculate delta
        mse += 0

    return mse / num_episodes_mse_calc


def get_policy(env: str):
    if env == gridworld.ENV:
        return gridworld.generate_random_gridworld_tabular_softmax_policy()

    if env == cartpole.ENV:
        return cartpole.generate_random_cartpole_policy()


def get_nth_order_fourier_basis(policy: np.ndarray, fourier_basis_n: int) -> list:
    return list(itertools.product(range(fourier_basis_n + 1), repeat=policy.shape[0]))
