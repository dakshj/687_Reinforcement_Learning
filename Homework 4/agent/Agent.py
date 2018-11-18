from abc import ABC, abstractmethod


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

    @abstractmethod
    def get_policy_from_q(self, q):
        pass

    @abstractmethod
    def take_action(self, action):
        pass

    @abstractmethod
    def gamma(self) -> float:
        pass
