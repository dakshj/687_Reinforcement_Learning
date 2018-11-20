from abc import ABC, abstractmethod

import numpy as np

from agent.agent import Agent


class TabularAgent(Agent, ABC):

    @staticmethod
    def is_tabular() -> bool:
        return True

    def init_q(self) -> np.ndarray:
        return np.zeros((self._num_states(), self._num_actions))

    @staticmethod
    @abstractmethod
    def _num_states():
        pass

    # Not needed for TabularAgents
    def get_phi(self) -> np.ndarray:
        pass
