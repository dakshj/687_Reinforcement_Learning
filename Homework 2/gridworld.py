import math
import operator
import random

import numpy as np

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


def tabular_softmax_policy(curr, table):
    # Convert coordinates to the 1-23 Gridworld range
    i = (curr[0] * ROWS) + (curr[1] + 1)

    # Subtract because of obstacles in GridWorld
    if i >= 13:
        i -= 1

    if i >= 18:
        i -= 1

    if i >= 23:
        i -= 1

    # Get row of probability values for each direction
    row = table[i]

    # Fetch a random direction based on the weighted probabilities of each direction
    return np.random.choice([UP, DOWN, LEFT, RIGHT], p=row)


def execute(episodes, policy_table):
    all_rewards = []

    for _ in range(episodes):

        curr = START
        step = -1
        reward = 0

        while True:
            step += 1

            if step > 15:
                all_rewards.append(reward)
                break

            direction = tabular_softmax_policy(curr, policy_table)

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
                all_rewards.append(reward)
                break

    all_rewards = np.array(all_rewards)

    return all_rewards
