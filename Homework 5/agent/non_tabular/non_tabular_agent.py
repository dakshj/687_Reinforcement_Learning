import itertools
import math
from abc import ABC, abstractmethod

import numpy as np

from agent.agent import Agent


class NonTabularAgent(Agent, ABC):

    def __init__(self, fourier_basis_order: int = None):
        super().__init__()

        self._fourier_arr = self._get_fourier_arr(
                fourier_basis_order=fourier_basis_order,
                state_dimension=self._get_state_dimension())
        self._num_features_phi = self._fourier_arr.shape[0]

    @staticmethod
    def _get_fourier_arr(fourier_basis_order: int, state_dimension: int) \
            -> np.ndarray:
        return np.array(list(itertools.product(range(fourier_basis_order + 1),
                repeat=state_dimension)))

    def get_phi(self, state) -> np.ndarray:
        return np.cos(math.pi * np.dot(self._fourier_arr,
                self._get_normalized_state(state)))

    @abstractmethod
    def _get_min_state_dimension_values(self) -> np.ndarray:
        pass

    @abstractmethod
    def _get_max_state_dimension_values(self) -> np.ndarray:
        pass

    @property
    def state(self) -> np.ndarray:
        return np.copy(self._state)

    def get_q_values_vector(self, state, weights: np.ndarray) -> np.ndarray:
        return np.dot(weights, self.get_phi(state))

    def _get_normalized_state(self, state):
        return (state - self._get_min_state_dimension_values()) / \
               (self._get_max_state_dimension_values() -
                self._get_min_state_dimension_values())

    @abstractmethod
    def _get_state_dimension(self) -> int:
        pass

    def init_e_v(self) -> np.ndarray:
        return np.zeros(self._num_features_phi)
