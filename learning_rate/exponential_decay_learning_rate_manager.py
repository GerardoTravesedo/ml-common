from learning_rate.learning_rate_manager import LearningRateManager
import math


class ExponentialDecayLearningRateManager(LearningRateManager):

    def __init__(self, initial_rate, decay):
        super().__init__()
        self.initial_learning_rate = initial_rate
        self.learning_rate = initial_rate
        self.decay = decay

    def get_learning_rate(self):
        return self.learning_rate

    def update_learning_rate(self, epoch):
        self.learning_rate = self.initial_learning_rate * math.exp(-self.decay * epoch)
        print("New learning rate is {}".format(self.learning_rate))
