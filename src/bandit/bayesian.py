import numpy as np

class Bayesian:

    def __init__(self, true_mean):
        self.true_mean = true_mean
        self.predicted_mean = 0 
        self.lambda_ = 1 # tau of estimation
        self.tau = 1 # 1 over sigma original distribution
        self.sum = 0 

    def pull(self):
        return np.random.randn() + self.true_mean
    
    def sample(self):
        return np.random.randn() / np.sqrt(self.lambda_) + self.predicted_mean

    def update(self, value):
        # we can update at each step from lambda = lambda_0 + tau * N
        self.lambda_ += self.tau
        # update observation
        self.sum += value
        # update mean form m = m_0*lambda_0 + tau * sum / lamba_0 + tau * N stepwise!
        self.predicted_mean = self.tau * self.sum / self.lambda_
