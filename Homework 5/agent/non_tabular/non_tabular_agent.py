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
            self._state_values_per_tile = np.divide(
                    np.subtract(self._get_max_state_dimension_values(),
                            self._get_min_state_dimension_values()),
                    self._tiles_per_tiling - 1)

    def init_weights(self) -> np.ndarray:
        if self._is_fourier_basis():
            return np.ones((self._num_actions, self._num_features_phi))

        elif self._is_tile_coding():
            return np.random.random((self._num_actions,
                                     self._num_features_tile_coding))

    @staticmethod
    def _get_fourier_arr(fourier_basis_order: int, state_dimension: int) \
            -> np.ndarray:
        return np.array(list(itertools.product(range(fourier_basis_order + 1),
                repeat=state_dimension)))

    def get_phi(self, state) -> np.ndarray:
        if self._is_fourier_basis():
            return np.cos(math.pi * np.dot(self._fourier_arr,
                    self.get_normalized_state(state)))
        elif self._is_tile_coding():
            return self._get_tile_coding_features(state)

    def _get_tile_coding_features(self, state):
        state_norm = self.get_normalized_state(state)
        feature_index_values = np.zeros(self._tilings)
        temp_array = np.zeros([self._tilings, self._get_state_dimension()])
        for i in range(self._tilings):
            for j in range(self._get_state_dimension()):
                temp_array[i, j] = int(state_norm[j] /
                                       self._state_values_per_tile[j] + i /
                                       self._tilings)
        for i in range(1, self._get_state_dimension()):
            temp_array[:, i] *= self._tiles_per_tiling ** i
        for i in range(self._tilings):
            feature_index_values[i] = (i * (self._tiles_per_tiling **
                                            self._get_state_dimension()) +
                                       sum(temp_array[i, :]))
        return feature_index_values

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

    @property
    def state(self) -> np.ndarray:
        return np.copy(self._state)

    def get_q_values_vector(self, state, q_or_weights: np.ndarray) -> np.ndarray:
        if self._is_fourier_basis():
            return np.dot(q_or_weights, self.get_phi(state))
        elif self._is_tile_coding():
            return np.array(
                    [self._compute_action_value_tile_coding(state,
                            q_or_weights, action_index)
                     for action_index in range(self._num_actions)])

    def _compute_action_value_tile_coding(self, state, weights: np.ndarray,
                                          action_index: int):
        action_value = 0
        phi = self.get_phi(state)
        for feature_index in phi:
            feature_index = min(feature_index, weights.shape[1] - 1)
            action_value += weights[action_index][int(feature_index)]
        return action_value

    def _get_hot_vector_tile_coding(self, features: np.ndarray):
        result = np.zeros(self._num_features_tile_coding)
        for feature_index in features:
            feature_index = min(feature_index, result.shape[0] - 1)
            result[int(feature_index)] = 1
        return result

    def get_features_for_weight_update(self, features: np.ndarray):
        if self._is_fourier_basis():
            return features
        elif self._is_tile_coding():
            return self._get_hot_vector_tile_coding(features=features)

    def get_normalized_state(self, state):
        return (state - self._get_min_state_dimension_values()) / \
               (self._get_max_state_dimension_values() -
                self._get_min_state_dimension_values())
