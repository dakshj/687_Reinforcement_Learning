import math

import numpy as np

from agent.non_tabular.non_tabular_agent import NonTabularAgent

ENV = 'cartpole'

GAMMA = 1.

MOTOR_FORCE = 10

G = 9.8

MASS_CART = 1
MASS_POLE = 0.1

POLE_HALF_LENGTH = 0.5

TIME_STEP_ACTUAL = 0.02
MAX_TIME_STEPS = 1010

REWARD_POLE_UP = 1

X_MIN, X_MAX = -3, 3
V_MIN, V_MAX = -10, 10
ANGLE_MIN, ANGLE_MAX = -math.pi / 2, math.pi / 2
THETA_DOT_MIN, THETA_DOT_MAX = -math.pi, math.pi

IDX_X, IDX_V, IDX_THETA, IDX_THETA_DOT = 0, 1, 2, 3

STATE_DIMENSION = 4


def calculate_accelerations(state: np.ndarray, action: int) -> (float, float):
    theta = state[IDX_THETA]
    theta_dot = state[IDX_THETA_DOT]

    sin0 = math.sin(theta)
    cos0 = math.cos(theta)

    num = (G * sin0) + (
            cos0 *
            (
                    -action -
                    (MASS_POLE * POLE_HALF_LENGTH * (theta_dot ** 2) * sin0)
            ) / (MASS_CART + MASS_POLE)
    )

    denom = POLE_HALF_LENGTH * ((4 / 3) - (
            (MASS_POLE * (cos0 ** 2)) /
            (MASS_CART + MASS_POLE)
    ))

    theta_dot_dot = num / denom

    x_dot_dot = (
                        action + (MASS_POLE * POLE_HALF_LENGTH * (
                        ((theta_dot ** 2) * sin0) - (theta_dot_dot * cos0))
                                  )
                ) / (MASS_CART + MASS_POLE)

    return theta_dot_dot, x_dot_dot


class CartPole(NonTabularAgent):

    def has_terminated(self) -> bool:
        pole_down = not (ANGLE_MIN <= self.state[IDX_THETA] <= ANGLE_MAX)
        cart_outside = not (X_MIN < self.state[IDX_X] < X_MAX)
        reached_max_time = self._time_step >= MAX_TIME_STEPS

        return pole_down or cart_outside or reached_max_time

    def _update_state_from_action(self, action):
        theta_dot_dot, x_dot_dot = calculate_accelerations(self.state, action)

        self._state[IDX_X] += (TIME_STEP_ACTUAL * self._state[IDX_V])
        self._state[IDX_V] += (TIME_STEP_ACTUAL * x_dot_dot)
        self._state[IDX_THETA] += (TIME_STEP_ACTUAL * self._state[IDX_THETA_DOT])
        self._state[IDX_THETA_DOT] += (TIME_STEP_ACTUAL * theta_dot_dot)

    def _get_initial_state(self) -> np.ndarray:
        return np.zeros(STATE_DIMENSION)

    def _get_state_dimension(self) -> int:
        return STATE_DIMENSION

    def _get_current_reward(self) -> float:
        if not self.has_terminated():
            return REWARD_POLE_UP

        return 0

    @property
    def gamma(self) -> float:
        return GAMMA

    @staticmethod
    def _get_actions_list() -> list:
        return [MOTOR_FORCE, -MOTOR_FORCE]

    def _get_min_state_dimension_values(self) -> np.ndarray:
        return np.array([X_MIN, V_MIN, ANGLE_MIN, THETA_DOT_MIN])

    def _get_max_state_dimension_values(self) -> np.ndarray:
        return np.array([X_MAX, V_MAX, ANGLE_MAX, THETA_DOT_MAX])
