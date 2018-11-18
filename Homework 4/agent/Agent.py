from abc import ABC, abstractmethod
from itertools import groupby
from operator import itemgetter


class Agent(ABC):

    def __init__(self):
        self.time_step = None
        self.state = None
        self.returns = None
        self.reset_for_new_episode()

    def reset_for_new_episode(self):
        self.time_step = 0
        self.state = self.__get_initial_state()
        self.returns = 0.

    @abstractmethod
    def env(self) -> str:
        pass

    @abstractmethod
    def is_tabular(self) -> bool:
        pass

    @abstractmethod
    def has_terminated(self) -> bool:
        pass

    @abstractmethod
    def __get_initial_state(self):
        pass

    @abstractmethod
    def get_action(self, policy):
        pass

    @staticmethod
    def get_policy_from_q(q) -> dict:
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

    @abstractmethod
    def take_action(self, action):
        pass

    @abstractmethod
    def gamma(self) -> float:
        pass
