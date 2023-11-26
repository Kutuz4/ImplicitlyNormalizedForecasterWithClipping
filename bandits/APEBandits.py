from bandits.AbstractMAB import AbstractMABSolver
import numpy as np
from scipy.stats import  gamma, uniform
class APEBandits(AbstractMABSolver):

    def __init__(self, N, alpha, sigma, warming_K=100, permutation=gamma(a=1), c=0.5):
        super().__init__(N, alpha, sigma)
        self.permutation = permutation
        self.p = 1 + alpha
        self.warming_K = warming_K
        self.c = c

        self.b = (2 * ((2 - self.p) / (self.p - 1)) ** (1 - 2 / self.p) +  ((2 - self.p) / (self.p - 1)) ** (2 - 2 / self.p)) ** (- self.p / 2)
        self.nums = np.zeros(self.N)

    def psi(self, x):
        if(x >= 0):
            return np.log(self.b * np.abs(x) ** self.p + x + 1)
        return -np.log(self.b * np.abs(x) ** self.p - x + 1)
    
    def choose_arm(self):
        if self.current_step <= self.warming_K:
            probs = np.ones((self.N)) * (1/self.N)
            arm = np.random.choice(self.N, p=probs)
        else:
            beta = self.c / (1e-6 + self.nums) ** (1 - 1. / self.p)
            G = self.permutation.ppf(uniform.rvs(size=self.N))
            r_hat = np.zeros(self.N)
            for j, arm in enumerate(self.arms_seq):
                r_hat[arm] += beta[arm] * self.psi(-self.losses[j] / (self.c * (1e-6 + self.nums[arm]) ** (1. / self.p)))
            rewards = r_hat + beta * G
            arm = np.argmax(rewards)
            probs = np.zeros(self.N)
            probs[arm] = 1.0
        return arm, probs
    
    def update_solver(self, loss, arm, probs):
        self.nums[arm] += 1
