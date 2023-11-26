from bandits.AbstractMAB import AbstractMABSolver
import numpy as np
from tools import SumRootsSolver

class OMDTsalisEntropyBandits(AbstractMABSolver):

    def __init__(self, N, alpha, sigma, T=5000):
        super().__init__(N, alpha, sigma)
        self.lambd = T ** (1/(1 + alpha)) * (2 * alpha / (1 - alpha)) ** (2 / (1 + alpha)) * sigma / (8 * N) ** (1/(1 + alpha))
        self.solverBeta = SumRootsSolver(self.N)
        self.probs = np.ones((self.N)) / self.N
        self.T = T

    def choose_arm(self):
        self.probs /= self.probs.sum()
        return np.random.choice(self.N, p=self.probs), self.probs
    
    def update_solver(self, loss, arm, probs):
        mu = np.sqrt(2) / np.sqrt(self.T * self.lambd ** (1 - self.alpha) * self.sigma ** (1 + self.alpha))
        grad = np.zeros((self.N))
        grad[arm] = min(loss, self.lambd) / probs[arm]
        beta_from_solver = self.solverBeta.solve(1/np.sqrt(probs) + mu * grad)
        self.probs = (beta_from_solver + 1/np.sqrt(probs) + mu * grad) ** (-2)