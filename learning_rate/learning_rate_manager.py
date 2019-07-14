from abc import ABC, abstractmethod


class LearningRateManager(ABC):

    @abstractmethod
    def get_learning_rate(self):
        pass
