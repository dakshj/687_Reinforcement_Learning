import itertools
import math
from abc import ABC

import numpy as np

from agent.agent import Agent


class NonTabularAgent(Agent, ABC):

    def __init__(self, epsilon: float, fourier_basis_order: int):
        super().__init__(epsilon)

        self._fourier_arr = self._get_fourier_arr(
                fourier_basis_order=fourier_basis_order,
                state_dimension=self._get_state_dimension()
        )

        self._num_features_phi = self._fourier_arr.shape[0]

    @staticmethod
    def is_tabular() -> bool:
        return False

    def init_weights(self):
        return np.zeros((self._num_actions, self._num_features_phi))

    @staticmethod
    def _get_fourier_arr(fourier_basis_order: int, state_dimension: int) \
            -> np.ndarray:
        return np.array(list(itertools.product(range(fourier_basis_order + 1),
                repeat=state_dimension)))

    def get_phi(self) -> np.ndarray:
        return np.cos(math.pi * np.dot(self._fourier_arr, self._state))

    # Not needed for NonTabularAgents
    @staticmethod
    def get_state_index(state):
        pass
