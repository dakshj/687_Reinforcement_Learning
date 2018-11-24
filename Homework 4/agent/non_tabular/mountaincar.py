import math

import numpy as np

from agent.non_tabular.non_tabular_agent import NonTabularAgent

ENV = 'mountaincar'

GAMMA = 1.

STATE_DIMENSION = 2

REWARD_NOT_REACHED_GOAL = -1

FORWARD, NEUTRAL, REVERSE = 1, 0, -1

IDX_X, IDX_V = 0, 1

X_BOUND_LOW, X_BOUND_HIGH = -1.2, 0.5

V_BOUND_LOW, V_BOUND_HIGH = -0.07, 0.07


class MountainCar(NonTabularAgent):

    def has_terminated(self) -> bool:
        return self._state[IDX_X] >= X_BOUND_HIGH

    def _update_state_from_action(self, action):
        x, v = self._state

        v += 0.001 * action - 0.0025 * math.cos(3 * x)

        # Bound velocity to be within limits
        if v < V_BOUND_LOW:
            v = V_BOUND_LOW
        elif v > V_BOUND_HIGH:
            v = V_BOUND_HIGH

        x += v

        # Bound position to be within limits
        if x < X_BOUND_LOW:
            x = X_BOUND_LOW
            v = 0

        elif x > X_BOUND_HIGH:
            x = X_BOUND_HIGH
            v = 0

        self._state = np.array([x, v])

    def _get_initial_state(self) -> np.ndarray:
        return np.array([-0.5, 0])

    def _get_state_dimension(self) -> int:
        return STATE_DIMENSION

    def _get_current_reward(self) -> float:
        if not self.has_terminated():
            return REWARD_NOT_REACHED_GOAL

        return 0

    @property
    def gamma(self) -> float:
        return GAMMA

    @staticmethod
    def _get_actions_list() -> list:
        return [FORWARD, NEUTRAL, REVERSE]