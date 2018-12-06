from abc import ABC, abstractmethod

import numpy as np

from agent.agent import Agent


class TabularAgent(Agent, ABC):

    def init_weights(self) -> np.ndarray:
        return np.random.random((self._num_states(), self._num_actions))

    def init_weights_actor_critic(self) -> np.ndarray:
        return np.random.random(self._num_states())

    @staticmethod
    @abstractmethod
    def _num_states():
        pass

    @property
    def state(self):
        return self._state

    def get_q_values_vector(self, state, weights: np.ndarray) -> np.ndarray:
        return weights[self.get_state_index(state)]

    @staticmethod
    @abstractmethod
    def get_state_index(state) -> int:
        pass

    def init_e_v(self) -> np.ndarray:
        return np.zeros(self._num_states())
