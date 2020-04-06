import numpy as np


class Bandit:
    def __init__(self, true_avg):
        self.true_avg = true_avg
        self.times_pulled = 0
        self.current_avg = 0

    def pull(self):
        self.times_pulled += 1
        value = self.true_avg * np.random.random_sample()
        #print(f'{value} was rolled.')
        return value

    def update_avg(self, value):
        one_by_n = 1 / self.times_pulled
        self.current_avg = (1 - one_by_n) * self.current_avg + one_by_n * value
