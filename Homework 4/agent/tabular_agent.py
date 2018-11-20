from abc import ABC, abstractmethod

import numpy as np

from agent.agent import Agent


class TabularAgent(Agent, ABC):

    def init_q(self) -> np.ndarray:
        return np.zeros((self._num_states(), self._num_actions))

    @staticmethod
    @abstractmethod
    def _num_states():
        pass

    @staticmethod
    @abstractmethod
    def get_state_index(state):
        pass
