import math
import random
from abc import ABC, abstractmethod

import numpy as np


class Agent(ABC):

    def __init__(self):
        self._time_step = None
        self._state = None
        self._returns = None
        self.reset_for_new_episode(epsilon=None)

        self._num_actions = len(self._get_actions_list())

        self._epsilon = None

    @property
    def state(self):
        return self._state

    @property
    def returns(self):
        return self._returns

    @property
    def time_step(self):
        return self._time_step

    def reset_for_new_episode(self, epsilon):
        self._time_step = 0
        self._state = self._get_initial_state()
        self._returns = 0.
        self._epsilon = epsilon

    @abstractmethod
    def has_terminated(self) -> bool:
        pass

    @staticmethod
    @abstractmethod
    def is_tabular() -> bool:
        pass

    def _update_returns(self):
        self._returns += math.pow(self.gamma, self._time_step) * \
                         self._get_current_reward()

    @abstractmethod
    def _update_state_from_action(self, action):
        pass

    @abstractmethod
    def _get_initial_state(self):
        pass

    @abstractmethod
    def _get_state_dimension(self):
        pass

    def get_action(self, q_or_weights: np.ndarray):
        random_no = random.random()

        pick_greedy_prob = 1 - self._epsilon + self._epsilon / self._num_actions

        # Take random action
        if random_no >= pick_greedy_prob:
            return np.random.choice(self._get_actions_list())

        # Choose best action from derived policy
        if self.is_tabular():
            q_values = q_or_weights[self.get_state_index(self._state)]
        else:
            q_values = np.dot(q_or_weights, self.get_phi())

        return self._get_actions_list()[int(np.argmax(q_values))]

    @staticmethod
    @abstractmethod
    def _get_actions_list() -> list:
        pass

    @abstractmethod
    def _get_current_reward(self) -> float:
        pass

    def take_action(self, action):
        self._update_state_from_action(action)
        self._update_returns()
        self._time_step += 1

        return self._get_current_reward(), self._state

    @property
    @abstractmethod
    def gamma(self) -> float:
        pass

    def get_action_index(self, action) -> int:
        return self._get_actions_list().index(action)

    @staticmethod
    @abstractmethod
    def get_state_index(state):
        pass

    @abstractmethod
    def get_phi(self) -> np.ndarray:
        pass
