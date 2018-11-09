import math
import operator
import random

import numpy as np

ENV = 'gridworld'

GAMMA = 0.9

ROWS, COLS = 5, 5

WALLS = [(2, 2), (3, 2)]
WATER = (4, 2)

START = (0, 0)
GOAL = (4, 4)

REWARD_GOAL = 10
REWARD_WATERS = -10

UP, DOWN, LEFT, RIGHT, STAY = 'AU', 'AD', 'AL', 'AR', 'STAY'

DIRECTIONS = {
    UP: (-1, 0),
    DOWN: (1, 0),
    LEFT: (0, -1),
    RIGHT: (0, 1),
    STAY: (0, 0)
}

PROB_ACTUAL = 0.8
PROB_LEFT = 0.85
PROB_RIGHT = 0.9
PROB_STAY = 1.0

VEERS = {
    UP: {
        LEFT: DIRECTIONS[LEFT], RIGHT: DIRECTIONS[RIGHT]
    },
    DOWN: {
        LEFT: DIRECTIONS[RIGHT], RIGHT: DIRECTIONS[LEFT]
    },
    LEFT: {
        LEFT: DIRECTIONS[DOWN], RIGHT: DIRECTIONS[UP]
    },
    RIGHT: {
        LEFT: DIRECTIONS[UP], RIGHT: DIRECTIONS[DOWN]
    },
}

MAX_ALLOWABLE_STEPS = 15


def get_direction_from_uniform_random_policy():
    return random.choice([UP, DOWN, LEFT, RIGHT])


def execute(episodes):
    all_returns = []

    for episode in range(episodes):

        state = START
        time_step = -1
        returns = 0

        while True:
            time_step += 1

            direction = get_direction_from_uniform_random_policy()

            direction_increment_coordinates = None

            rand_actual_no = random.random()

            if rand_actual_no < PROB_ACTUAL:
                direction_increment_coordinates = DIRECTIONS[direction]
            elif PROB_ACTUAL <= rand_actual_no < PROB_LEFT:
                direction_increment_coordinates = VEERS[direction][LEFT]
            elif PROB_LEFT <= rand_actual_no < PROB_RIGHT:
                direction_increment_coordinates = VEERS[direction][RIGHT]
            elif PROB_RIGHT <= rand_actual_no < PROB_STAY:
                direction_increment_coordinates = DIRECTIONS[STAY]

            temp_state = state

            temp_state = tuple(map(operator.add, temp_state,
                direction_increment_coordinates))

            if temp_state not in WALLS and \
                    (0 <= temp_state[0] < ROWS) and \
                    (0 <= temp_state[1] < COLS):
                state = temp_state

            reward = 0

            if state == WATER:
                reward = math.pow(GAMMA, time_step) * REWARD_WATERS

            elif state == GOAL:
                reward = math.pow(GAMMA, time_step) * REWARD_GOAL

            returns += reward

            yield time_step, episode, None, reward, GAMMA

            if state == GOAL or time_step >= MAX_ALLOWABLE_STEPS:
                all_returns.append(returns)
                break

    return np.array(all_returns)


def generate_random_gridworld_tabular_softmax_policy():
    return np.random.uniform(0, 1, (92,))


def get_v_vector() -> np.ndarray:
    result = np.zeros(shape=(5, 5))

    # TODO Calculate *expected* v values for each of the 23 states

    return result
