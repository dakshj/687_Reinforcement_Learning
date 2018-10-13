import math

import numpy as np

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


def get_force_from_next_action(policy: np.ndarray, curr_state: np.ndarray) -> int:
    result = np.dot(policy, curr_state) + curr_state[IDX_THETA]

    # Only two actions: {Apply Force Left} or {Apply Force Right}
    if result > 0:
        return MOTOR_FORCE
    else:
        return -MOTOR_FORCE


def update_state(state: np.ndarray, theta_dot_dot: float, x_dot_dot: float):
    state[IDX_X] = state[IDX_X] + (TIME_STEP * state[IDX_V])
    state[IDX_V] = state[IDX_V] + (TIME_STEP * x_dot_dot)
    state[IDX_THETA] = state[IDX_THETA] + (TIME_STEP * state[IDX_THETA_DOT])
    state[IDX_THETA_DOT] = state[IDX_THETA_DOT] + (TIME_STEP * theta_dot_dot)


def execute(episodes: int, policy: np.ndarray) -> list:
    all_rewards = []

    for _ in range(episodes):
        state = np.zeros((4,))

        reward = 0

        # Step over every 0.02s until the 20.2s limit
        for _ in np.arange(0., TIME_TOTAL, TIME_STEP):
            if not (ANGLE_MIN <= state[IDX_THETA] <= ANGLE_MAX):
                # Pole has fallen down
                reward += REWARD_POLE_DOWN
                break

            if not (X_MIN < state[IDX_X] < X_MAX):
                break

            # Pole is still up
            reward += REWARD_POLE_UP

            force_applied_to_cart = get_force_from_next_action(policy, state)

            theta = state[IDX_THETA]

            num = (G * math.sin(theta)) + \
                  (
                      math.cos(theta) *
                      (
                          -force_applied_to_cart -
                          (MASS_POLE * POLE_HALF_LENGTH * (theta ** 2) * math.sin(theta))
                      )
                  )

            denom = POLE_HALF_LENGTH * ((4 / 3) - (
                (MASS_POLE * (math.cos(theta) ** 2)) /
                (MASS_CART + MASS_POLE)
            ))

            theta_dot_dot = num / denom

            x_dot_dot = (force_applied_to_cart + (
                MASS_POLE * POLE_HALF_LENGTH * (
                    ((theta ** 2) * math.sin(theta)) - (theta_dot_dot * math.cos(theta)))
            )) \
                        / (MASS_CART + MASS_POLE)

            update_state(state, theta_dot_dot, x_dot_dot)

            # End for loop

        all_rewards.append(reward)

    # Return rewards of all episodes
    return all_rewards


if __name__ == '__main__':
    execute(episodes=1, policy=np.random.uniform(-10, 10, (4,)))