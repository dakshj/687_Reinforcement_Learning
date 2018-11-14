import math
import random

import numpy as np

ENV = 'cartpole'

GAMMA = 1

ANGLE_MAX = math.pi / 2
ANGLE_MIN = -math.pi / 2
ANGLE_START = 0

MOTOR_FORCE = 10

G = 9.8

MASS_CART = 1
MASS_POLE = 0.1

POLE_HALF_LENGTH = 0.5

TIME_STEP = 0.02
TIME_TOTAL = 20.2

REWARD_POLE_UP = 1
REWARD_POLE_DOWN = 0

ACTION_LEFT = 'left'
ACTION_RIGHT = 'right'

X_MAX = 3
X_MIN = -3

IDX_X, IDX_V, IDX_THETA, IDX_THETA_DOT = 0, 1, 2, 3


def get_force_from_random_policy() -> int:
    return MOTOR_FORCE if bool(random.getrandbits(1)) else -MOTOR_FORCE


def update_state(state: np.ndarray, theta_dot_dot: float, x_dot_dot: float):
    state[IDX_X] += (TIME_STEP * state[IDX_V])
    state[IDX_V] += (TIME_STEP * x_dot_dot)
    state[IDX_THETA] += (TIME_STEP * state[IDX_THETA_DOT])
    state[IDX_THETA_DOT] += (TIME_STEP * theta_dot_dot)


def calculate_accelerations(state: np.ndarray) -> (float, float):
    force_applied_to_cart = get_force_from_random_policy()

    theta = state[IDX_THETA]
    theta_dot = state[IDX_THETA_DOT]

    sin0 = math.sin(theta)
    cos0 = math.cos(theta)

    num = (G * sin0) + (
            cos0 *
            (
                    -force_applied_to_cart -
                    (MASS_POLE * POLE_HALF_LENGTH * (theta_dot ** 2) * sin0)
            ) / (MASS_CART + MASS_POLE)
    )

    denom = POLE_HALF_LENGTH * ((4 / 3) - (
            (MASS_POLE * (cos0 ** 2)) /
            (MASS_CART + MASS_POLE)
    ))

    theta_dot_dot = num / denom

    x_dot_dot = (
                        force_applied_to_cart + (MASS_POLE * POLE_HALF_LENGTH * (
                        ((theta_dot ** 2) * sin0) - (theta_dot_dot * cos0))
                                                 )
                ) / (MASS_CART + MASS_POLE)

    return theta_dot_dot, x_dot_dot


def execute(episodes: int) -> list:
    all_rewards = []

    for episode in range(episodes):
        state = np.zeros((4,))

        reward = 0

        # Step over every 0.02s until the 20.2s limit
        for time_step in range(int(TIME_TOTAL / TIME_STEP)):
            if not (ANGLE_MIN <= state[IDX_THETA] <= ANGLE_MAX):
                # Pole has fallen down
                reward += REWARD_POLE_DOWN
                break

            if not (X_MIN < state[IDX_X] < X_MAX):
                break

            yield time_step, episode, state, REWARD_POLE_UP, GAMMA

            # Pole is still up
            reward += REWARD_POLE_UP

            theta_dot_dot, x_dot_dot = calculate_accelerations(state)

            update_state(state, theta_dot_dot, x_dot_dot)

            # End time steps loop

        all_rewards.append(reward)

        # End episodes loop

    # Return rewards of all episodes
    return all_rewards
