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


def uniform_random_policy():
    return random.choice([UP, DOWN, LEFT, RIGHT])


def execute(episodes):
    all_rewards = []

    for episode in range(episodes):

        curr = START
        step = -1
        reward = 0

        while True:
            step += 1

            direction = uniform_random_policy()

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

            curr_temp = curr

            curr_temp = tuple(map(operator.add, curr_temp, direction_increment_coordinates))

            if curr_temp not in WALLS and (0 <= curr_temp[0] < ROWS) and (0 <= curr_temp[1] < COLS):
                curr = curr_temp

            if curr == WATER:
                reward += math.pow(GAMMA, step) * REWARD_WATERS

            elif curr == GOAL:
                reward += math.pow(GAMMA, step) * REWARD_GOAL

            # TODO yield values based on what is required by td.py
            yield step, episode

            if curr == GOAL or step >= 15:
                all_rewards.append(reward)
                break

    all_rewards = np.array(all_rewards)

    return all_rewards


def generate_random_gridworld_tabular_softmax_policy():
    return np.random.uniform(0, 1, (92,))
