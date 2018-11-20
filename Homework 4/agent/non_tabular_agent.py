import itertools
import math
from abc import ABC

import numpy as np

from agent.agent import Agent


class NonTabularAgent(Agent, ABC):

    def __init__(self, epsilon: float, fourier_basis_order: int):
        super().__init__(self, epsilon)

        self.fourier_arr = NonTabularAgent._get_fourier_arr(
                fourier_basis_order=fourier_basis_order,
                state_vector_length=self.get_state_vector_length()
        )

        self.num_features_phi = self.fourier_arr.shape[0]

    def init_weights(self):
        return np.zeros((self._num_actions, self.num_features_phi))

    @staticmethod
    def _get_fourier_arr(fourier_basis_order: int, state_vector_length: int) \
            -> np.ndarray:
        return np.array(list(itertools.product(range(fourier_basis_order + 1),
                repeat=state_vector_length)))

    def get_phi(self) -> np.ndarray:
        return np.cos(math.pi * np.dot(self.fourier_arr, self._state))
