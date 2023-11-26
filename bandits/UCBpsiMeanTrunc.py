from bandits.AbstractMAB import AbstractMABSolver
import numpy as np


class UCBpsiMeanTrunc(AbstractMABSolver):

    def __init__(self, N, alpha, sigma, c, delta=0.5):
        super().__init__(N, alpha, sigma)
        self.c = c
        self.delta = delta
        self.arms_stats = [[] for _ in range(self.N)]

    def calculate_truncated_mean(self, stats):
        if len(stats) == 0:
            return -1 * np.inf
        return sum([x * int(x < (self.sigma * len(stats) / np.log(1/self.delta)) ** (1/(1+self.alpha))) for x in stats]) / len(stats)
    
    def calculate_additive_part(self, stats):
        return self.sigma ** (1/(1 + self.alpha)) * (self.c * np.log(self.current_step ** 2) / (len(stats))) ** (self.alpha/(1 + self.alpha))
    
    def choose_arm(self):
        B = [self.calculate_additive_part(x) + self.calculate_truncated_mean(x) if len(x) != 0 else -np.inf for x in self.arms_stats]
        probs = np.zeros(self.N)
        arm = np.argmin(B)
        probs[arm] = 1.0
        return arm, probs
    
    def update_solver(self, loss, arm, probs):
        self.arms_stats[arm].append(loss)