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
            self._num_features_tile_coding = \
                self._tilings * (self._tiles_per_tiling ** self._get_state_dimension())

    @staticmethod
    def is_tabular() -> bool:
        return False

    def init_weights(self) -> np.ndarray:
        if self._is_fourier_basis():
            return np.random.random((self._num_actions, self._num_features_phi))

        elif self._is_tile_coding():
            return np.random.random((self._num_actions, self._num_features_tile_coding))

    def get_tile_coding_features(self, state):
        state -= self._get_min_state_dimension_values()
        tileIndices = np.zeros(self._tilings)
        matrix = np.zeros([self._tilings, self._get_state_dimension()])
        for i in range(self._tilings):
            for j in range(self._get_state_dimension()):
                matrix[i, j] = int(state[j] / self.tileSize[j] \
                                    + i / self.numTilings)
        for i in range(1, self.dim):
            matrix[:, i] *= self.tilesPerTiling ** i
        for i in range(self.numTilings):
            tileIndices[i] = (i * (self.tilesPerTiling ** self.dim) \
                              + sum(matrix[i, :]))
        return tileIndices

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
