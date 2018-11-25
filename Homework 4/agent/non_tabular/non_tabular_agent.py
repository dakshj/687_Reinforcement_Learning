import itertools
import math
from abc import ABC, abstractmethod

import numpy as np

from agent.agent import Agent


class NonTabularAgent(Agent, ABC):
    FOURIER_BASIS = 'fourier_basis'
    TILE_CODING = 'tile_coding'

    def __init__(self, function_approximation_method=FOURIER_BASIS,
                 fourier_basis_order: int = None,
                 tilings: int = None, tiles_per_tiling: int = None):
        super().__init__()

        self._func_approx_method = function_approximation_method

        if self._is_fourier_basis():
            self._fourier_arr = self._get_fourier_arr(
                    fourier_basis_order=fourier_basis_order,
                    state_dimension=self._get_state_dimension())
            self._num_features_phi = self._fourier_arr.shape[0]

        elif self._is_tile_coding():
            self._tilings = tilings
            self._tiles_per_tiling = tiles_per_tiling

    @staticmethod
    def is_tabular() -> bool:
        return False

    def init_weights(self) -> np.ndarray:
        if self._is_fourier_basis():
            return np.random.random((self._num_actions, self._num_features_phi))

        elif self._is_tile_coding():
            # TODO Return weights with dimensions that are correct for tile coding
            return None

    @staticmethod
    def _get_fourier_arr(fourier_basis_order: int, state_dimension: int) \
            -> np.ndarray:
        return np.array(list(itertools.product(range(fourier_basis_order + 1),
                repeat=state_dimension)))

    def get_phi(self) -> np.ndarray:
        return np.cos(math.pi * np.dot(self._fourier_arr, self.state))

    # Not needed for NonTabularAgents
    @staticmethod
    def get_state_index(state) -> int:
        pass

    def _is_fourier_basis(self) -> bool:
        return self._func_approx_method is NonTabularAgent.FOURIER_BASIS

    def _is_tile_coding(self) -> bool:
        return self._func_approx_method is NonTabularAgent.TILE_CODING

    @abstractmethod
    def _get_min_state_dimension_values(self) -> np.ndarray:
        pass

    @abstractmethod
    def _get_max_state_dimension_values(self) -> np.ndarray:
        pass
