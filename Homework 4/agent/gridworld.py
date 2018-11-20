import operator
import random

import numpy as np

from agent.tabular_agent import TabularAgent

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


def tabular_softmax_policy(policy_table, state):
    # Convert coordinates to the 1-23 Gridworld range
    i = (state[0] * ROWS) + (state[1] + 1)

    # Subtract because of obstacles in GridWorld
    if i >= 13:
        i -= 1

    if i >= 18:
        i -= 1

    if i >= 23:
        i -= 1

    # Get row of probability values for each direction
    row = policy_table[i]

    # Fetch a random direction based on the weighted probabilities of each direction
    return np.random.choice(GridWorld.get_actions_list(), p=row)


class GridWorld(TabularAgent):

    def has_terminated(self) -> bool:
        return self.state == GOAL or self._time_step >= MAX_ALLOWABLE_STEPS

    def _update_state_from_action(self, action):
        direction_increment_coordinates = None

        rand_actual_no = random.random()

        if rand_actual_no < PROB_ACTUAL:
            direction_increment_coordinates = DIRECTIONS[action]
        elif PROB_ACTUAL <= rand_actual_no < PROB_LEFT:
            direction_increment_coordinates = VEERS[action][LEFT]
        elif PROB_LEFT <= rand_actual_no < PROB_RIGHT:
            direction_increment_coordinates = VEERS[action][RIGHT]
        elif PROB_RIGHT <= rand_actual_no < PROB_STAY:
            direction_increment_coordinates = DIRECTIONS[STAY]

        temp_state = self.state

        temp_state = tuple(map(operator.add, temp_state,
                direction_increment_coordinates))

        if temp_state not in WALLS and \
                (0 <= temp_state[0] < ROWS) and \
                (0 <= temp_state[1] < COLS):
            self.state = temp_state

    def get_initial_state(self):
        return START

    def get_state_vector_length(self) -> int:
        return len(self.get_initial_state())

    def _get_current_reward(self) -> float:
        if self.state == WATER:
            return REWARD_WATERS

        elif self.state == GOAL:
            return REWARD_GOAL

        return 0

    def gamma(self) -> float:
        return GAMMA

    @staticmethod
    def get_actions_list() -> list:
        return [UP, DOWN, LEFT, RIGHT]

    def _get_action_from_policy(self, policy):
        return tabular_softmax_policy(policy, self.state)
