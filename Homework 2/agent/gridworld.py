import math
import operator
import random

import numpy as np
from preconditions import preconditions

from util.softmax import softmax

ENV = 'gridworld'

GAMMA = 0.9

ROWS, COLS = 5, 5
GRID_SHAPE = (ROWS, COLS)

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


def tabular_softmax_policy(curr, table):
    # Convert coordinates to the 1-23 Gridworld range
    i = (curr[0] * ROWS) + (curr[1] + 1)

    # Subtract because of obstacles in GridWorld
    if i >= 14:
        i -= 1

    if i >= 18:
        i -= 1

    # 0-based value
    i -= 1

    # Get row of probability values for each direction
    row = table[i]

    # Fetch a random direction based on the weighted probabilities of each direction
    return np.random.choice([UP, DOWN, LEFT, RIGHT], p=row)


@preconditions(
        lambda policy_table: np.shape(policy_table) == (23, 4)
)
def execute(episodes, policy_table):
    all_returns = []

    for episode in range(episodes):

        state = START
        time_step = 0
        returns = 0
        reward = 0

        while True:
            if state == GOAL or time_step >= MAX_ALLOWABLE_STEPS:
                all_returns.append(returns)
                break

            direction = tabular_softmax_policy(state, policy_table)

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

            if state == WATER:
                reward = math.pow(GAMMA, time_step) * REWARD_WATERS

            elif state == GOAL:
                reward = math.pow(GAMMA, time_step) * REWARD_GOAL

            returns += reward
            time_step += 1

    return np.array(all_returns)


def generate_random_gridworld_tabular_softmax_policy():
    return np.random.uniform(0, 1, (92,))


def convert_theta_to_table(theta):
    return softmax(X=np.reshape(theta, (-1, 4)), theta=0.5, axis=1)
