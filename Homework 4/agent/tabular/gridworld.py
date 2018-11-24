import operator
import random

from agent.tabular.tabular_agent import TabularAgent

ENV = 'gridworld'

GAMMA = 0.9

TOTAL_STATES = 23

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

MAX_TIME_STEPS = 50


class GridWorld(TabularAgent):

    def has_terminated(self) -> bool:
        return self._state == GOAL or self._time_step >= MAX_TIME_STEPS

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

        temp_state = self._state

        temp_state = tuple(map(operator.add, temp_state,
                direction_increment_coordinates))

        if temp_state not in WALLS and \
                (0 <= temp_state[0] < ROWS) and \
                (0 <= temp_state[1] < COLS):
            self._state = temp_state

    def _get_initial_state(self):
        return START

    def _get_state_dimension(self) -> int:
        return len(self._get_initial_state())

    def _get_current_reward(self) -> float:
        if self._state == WATER:
            return REWARD_WATERS

        elif self._state == GOAL:
            return REWARD_GOAL

        return 0

    @property
    def gamma(self) -> float:
        return GAMMA

    @staticmethod
    def _get_actions_list() -> list:
        return [UP, DOWN, LEFT, RIGHT]

    @staticmethod
    def _num_states():
        return TOTAL_STATES

    @staticmethod
    def get_state_index(state) -> int:
        # Convert coordinates to the 1-23 Gridworld range
        index = (state[0] * ROWS) + (state[1] + 1)

        # Subtract because of obstacles in GridWorld
        if index >= 14:
            index -= 1

        if index >= 18:
            index -= 1

        # 0-based value
        index -= 1

        return index
