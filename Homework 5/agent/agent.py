import math
from abc import ABC, abstractmethod

import numpy as np

EPSILON_GREEDY = 'ep'
SOFTMAX = 's'


class Agent(ABC):

    def __init__(self):
        self._time_step = None
        self._state = None
        self._returns = None
        self.reset_for_new_episode(epsilon=None, sigma=None)

        self._num_actions = len(self._get_actions_list())

        self._epsilon = None
        self._sigma = None

    @property
    @abstractmethod
    def state(self):
        pass

    @property
    def returns(self) -> float:
        return self._returns

    @property
    def time_step(self) -> int:
        return self._time_step

    def reset_for_new_episode(self, epsilon=None, sigma=None):
        self._time_step = 0
        self._state = self._get_initial_state()
        self._returns = 0.
        self._epsilon = epsilon
        self._sigma = sigma

    @abstractmethod
    def has_terminated(self) -> bool:
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
    def get_q_values_vector(self, state, weights: np.ndarray) -> np.ndarray:
        pass

    def epsilon_greedy(self, weights):
        pi = np.ones(self._num_actions) * self._epsilon / self._num_actions

        q_values = self.get_q_values_vector(state=self.state, weights=weights)
        pi[int(np.argmax(q_values))] += 1 - self._epsilon

        return pi

    def softmax(self, weights):
        q_values = self.get_q_values_vector(state=self.state, weights=weights)
        pi = np.exp(self._sigma * q_values - np.max(self._sigma * q_values)) / \
             np.sum(np.exp(self._sigma * q_values - np.max(self._sigma * q_values)))

        return pi

    def get_action(self, weights: np.ndarray, action_selection_method: str):
        method = None
        if action_selection_method is EPSILON_GREEDY:
            method = self.epsilon_greedy
        elif action_selection_method is SOFTMAX:
            method = self.softmax

        return np.random.choice(self._get_actions_list(), p=method(weights=weights))

    def get_max_q_value(self, state, weights: np.ndarray) -> float:
        return np.max(self.get_q_values_vector(state=state, weights=weights))

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

        return self._get_current_reward(), self.state

    @property
    @abstractmethod
    def gamma(self) -> float:
        pass

    def get_action_index(self, action) -> int:
        return self._get_actions_list().index(action)

    def init_e_trace(self) -> np.ndarray:
        return np.zeros_like(self.init_weights())

    @abstractmethod
    def init_weights(self) -> np.ndarray:
        """
        For Actor-Critic, this will be used for `theta`, and not `weights`.
        """
        pass

    @abstractmethod
    def init_weights_actor_critic(self) -> np.ndarray:
        pass

    @abstractmethod
    def init_e_v(self) -> np.ndarray:
        pass

    def init_e_theta(self) -> np.ndarray:
        return self.init_e_trace()

    @property
    def num_actions(self):
        return self._num_actions
