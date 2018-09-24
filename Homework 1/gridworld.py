import math
import operator
import random

import numpy as np

NUM_EPISODES = 10000

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

OPTIMAL_POLICY = [
    [RIGHT, RIGHT, RIGHT, RIGHT, DOWN],
    [RIGHT, RIGHT, RIGHT, DOWN, DOWN],
    [UP, UP, STAY, DOWN, DOWN],
    [UP, UP, STAY, DOWN, DOWN],
    [UP, LEFT, RIGHT, RIGHT, STAY],
]


def uniform_random_policy(curr):
    rand_no = random.randint(0, 4)

    if rand_no == 0:
        direction = UP
    elif rand_no == 1:
        direction = DOWN
    elif rand_no == 2:
        direction = LEFT
    else:
        direction = RIGHT

    return direction


def optimal_policy(curr):
    return OPTIMAL_POLICY[curr[0]][curr[1]]


def execute(episodes=NUM_EPISODES, policy=uniform_random_policy, print_s19_given_s8=False):
    if policy == uniform_random_policy:
        print('~~ Uniform Random Policy ~~')
    elif policy == optimal_policy:
        print('~~ Optimal Policy ~~')

    if print_s19_given_s8:
        print('~~ With Calculation for "Pr(S19 = 21 | S8 = 18)" ~~')

    all_rewards = []

    s8_count = 0
    s19_count = 0

    for i in range(episodes):

        curr = START
        step = -1
        reward = 0

        s8_reached = False
        s19_reached = False

        while True:
            step += 1

            if step == 8 and curr == (3, 4):
                s8_reached = True

            if s8_reached and step == 19 and curr == WATER:
                s19_reached = True

            direction = policy(curr)

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

                if s8_reached:
                    s8_count += 1

                if s19_reached:
                    s19_count += 1

                break

    all_rewards = np.array(all_rewards)

    print('Mean = {}'.format(np.mean(all_rewards)))
    print('Standard Deviation = {}'.format(np.std(all_rewards)))
    print('Maximum = {}'.format(np.max(all_rewards)))
    print('Minimum = {}'.format(np.min(all_rewards)))

    if print_s19_given_s8:
        print('Pr(S19 = 21 | S8 = 18)  =  {}'.format(s19_count / s8_count))

    print('')


if __name__ == '__main__':
    run_part_1 = True
    run_part_3 = False
    run_part_4 = False

    if run_part_1:
        execute()

    if run_part_3:
        execute(policy=optimal_policy)

    if run_part_4:
        execute(print_s19_given_s8=True)
