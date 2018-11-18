from abc import abstractmethod

from agent import Agent


class NonTabularAgent(Agent):

    @staticmethod
    def is_tabular():
        return False

    @abstractmethod
    def init_w(self):
        pass
