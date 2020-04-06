import numpy as np


class Bandit:
    def __init__(self, true_mean):
        self.true_mean = true_mean
        self.times_pulled = 0
        self.current_mean = 0

    def pull(self):
        self.times_pulled += 1
        value = self.true_mean + np.random.randn()
        #print(f'{value} was rolled.')
        return value

    def update(self, value):
        one_by_n = 1 / self.times_pulled
        self.current_mean = (1 - one_by_n) * self.current_mean + one_by_n * value
