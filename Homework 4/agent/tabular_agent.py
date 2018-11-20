from abc import ABC
from collections import defaultdict
from itertools import groupby
from operator import itemgetter

from agent.agent import Agent


class TabularAgent(Agent, ABC):

    @staticmethod
    def init_q() -> dict:
        return defaultdict(float)

    @staticmethod
    def get_policy_from_q(q: dict) -> dict:
        """
        Chooses the best actions from the q dict, for each state

        :param q:
        :return: Policy dict of state to action
        """
        grouped = [(k, [x for _, x in group]) for k, group in groupby(q, itemgetter(0))]

        policy = {}

        for state, action_list in grouped:
            max_val = None
            max_action = None

            for action in action_list:
                if max_val is None:
                    max_val = q[(state, action)]
                    max_action = action

                if q[(state, action)] > max_val:
                    max_val = q[(state, action)]
                    max_action = action

            policy[state] = max_action

        return policy
