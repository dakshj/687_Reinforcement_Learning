import random

import numpy as np

from agent.non_tabular.non_tabular_agent import NonTabularAgent

ENV = 'mountaincar'

GAMMA = 1.

STATE_DIMENSION = 2

REWARD_NOT_REACHED_GOAL = -1

THROTTLE, IDLE, REVERSE = 1, 0, -1

IDX_X, IDX_V = 0, 1

X_BOUND_LOW = -1.2
X_BOUND_HIGH = 0.5

V_BOUND_LOW = -0.07
V_BOUND_HIGH = 0.07


class MountainCar(NonTabularAgent):

    def has_terminated(self) -> bool:
        # TODO
        pass

    def _update_state_from_action(self, action):
        # TODO
        pass

    def _get_initial_state(self) -> np.ndarray:
        return np.array([
            random.uniform(X_BOUND_LOW, X_BOUND_HIGH),
            random.uniform(V_BOUND_LOW, V_BOUND_HIGH)
        ])

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
        return [THROTTLE, IDLE, REVERSE]
