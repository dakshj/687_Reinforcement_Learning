import math

import numpy as np

from agent import NonTabularAgent

ENV = 'cartpole'

GAMMA = 1.

ANGLE_MAX = math.pi / 2
ANGLE_MIN = -math.pi / 2
ANGLE_START = 0

MOTOR_FORCE = 10

G = 9.8

MASS_CART = 1
MASS_POLE = 0.1

POLE_HALF_LENGTH = 0.5

TIME_STEP_ACTUAL = 0.02
MAX_TIME_STEPS = 1010

REWARD_POLE_UP = 1

ACTION_LEFT = 'left'
ACTION_RIGHT = 'right'

X_MAX = 3
X_MIN = -3

IDX_X, IDX_V, IDX_THETA, IDX_THETA_DOT = 0, 1, 2, 3


def calculate_accelerations(state: np.ndarray, action: int) -> (float, float):
    theta = state[IDX_THETA]
    theta_dot = state[IDX_THETA_DOT]

    sin0 = math.sin(theta)
    cos0 = math.cos(theta)

    num = (G * sin0) + (
            cos0 *
            (
                    -action -
                    (MASS_POLE * POLE_HALF_LENGTH * (theta_dot ** 2) * sin0)
            ) / (MASS_CART + MASS_POLE)
    )

    denom = POLE_HALF_LENGTH * ((4 / 3) - (
            (MASS_POLE * (cos0 ** 2)) /
            (MASS_CART + MASS_POLE)
    ))

    theta_dot_dot = num / denom

    x_dot_dot = (
                        action + (MASS_POLE * POLE_HALF_LENGTH * (
                        ((theta_dot ** 2) * sin0) - (theta_dot_dot * cos0))
                                  )
                ) / (MASS_CART + MASS_POLE)

    return theta_dot_dot, x_dot_dot


class CartPole(NonTabularAgent):

    @staticmethod
    def env() -> str:
        return ENV

    @staticmethod
    def __get_initial_state():
        return np.zeros((4,))

    @staticmethod
    def gamma() -> float:
        return GAMMA

    @staticmethod
    def init_w(self):
        # TODO
        pass

    def has_terminated(self) -> bool:
        pole_down = not (ANGLE_MIN <= self.state[IDX_THETA] <= ANGLE_MAX)
        cart_outside = not (X_MIN < self.state[IDX_X] < X_MAX)
        reached_max_time = self.time_step >= MAX_TIME_STEPS

        return pole_down or cart_outside or reached_max_time

    @staticmethod
    def get_initial_state():
        return np.zeros((4,))

    def get_action(self, policy):
        result = np.dot(policy, self.state) + self.state[IDX_THETA]

        # Only two actions: {Apply Force Left} or {Apply Force Right}
        if result > 0:
            return MOTOR_FORCE
        else:
            return -MOTOR_FORCE

    def take_action(self, action):
        self.__update_state_from_action(action)
        self.__update_returns()
        self.time_step += 1

        return self.__get_reward(), self.state

    def __update_returns(self):
        self.returns += math.pow(GAMMA, self.time_step) * self.__get_reward()

    def __get_reward(self):
        if not self.has_terminated():
            return REWARD_POLE_UP

        return 0

    def __update_state_from_action(self, action):
        theta_dot_dot, x_dot_dot = calculate_accelerations(self.state, action)

        self.state[IDX_X] += (TIME_STEP_ACTUAL * self.state[IDX_V])
        self.state[IDX_V] += (TIME_STEP_ACTUAL * x_dot_dot)
        self.state[IDX_THETA] += (TIME_STEP_ACTUAL * self.state[IDX_THETA_DOT])
        self.state[IDX_THETA_DOT] += (TIME_STEP_ACTUAL * theta_dot_dot)

    def calculate_accelerations(self, action: int) -> (float, float):
        theta = self.state[IDX_THETA]
        theta_dot = self.state[IDX_THETA_DOT]

        sin0 = math.sin(theta)
        cos0 = math.cos(theta)

        num = (G * sin0) + (
                cos0 *
                (
                        -action -
                        (MASS_POLE * POLE_HALF_LENGTH * (theta_dot ** 2) * sin0)
                ) / (MASS_CART + MASS_POLE)
        )

        denom = POLE_HALF_LENGTH * ((4 / 3) - (
                (MASS_POLE * (cos0 ** 2)) /
                (MASS_CART + MASS_POLE)
        ))

        theta_dot_dot = num / denom

        x_dot_dot = (
                            action + (MASS_POLE * POLE_HALF_LENGTH * (
                            ((theta_dot ** 2) * sin0) - (theta_dot_dot * cos0))
                                      )
                    ) / (MASS_CART + MASS_POLE)

        return theta_dot_dot, x_dot_dot
