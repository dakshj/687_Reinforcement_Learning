from abc import abstractmethod

from agent import Agent


class TabularAgent(Agent):

    @staticmethod
    def is_tabular():
        return True

    @abstractmethod
    def init_q(self) -> dict:
        pass
