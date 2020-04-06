import numpy as np

class Bayesian:

    def __init__(self, true_mean):
        self.true_mean = true_mean
        self.predicted_mean = 0
        self.lambda_ = 1
        self.tau = 1 # 1 over sigma
        self.sum = 0

    def pull(self):
        return np.random.randn() + self.true_mean
    
    def sample(self):
        return np.random.randn() / np.sqrt(self.lambda_) + self.predicted_mean

    def update(self, value):
        self.lambda_ += self.tau
        self.sum += value
        self.predicted_mean = self.tau * self.sum / self.lambda_
