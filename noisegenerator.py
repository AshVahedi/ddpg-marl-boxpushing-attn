import numpy as np

class OUNoise:
    def __init__(self, dim, mu=0.0, theta=0.15):
        self.mu = mu
        self.theta = theta
        self.state = np.ones(dim) * self.mu

    def reset(self):
        self.state = np.ones_like(self.state) * self.mu

    def sample(self,sigma):
        dx = self.theta * (self.mu - self.state) + sigma * np.random.randn(len(self.state))
        self.state += dx
        return self.state
