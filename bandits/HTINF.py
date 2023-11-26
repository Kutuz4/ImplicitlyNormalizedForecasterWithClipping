from bandits.AbstractMAB import AbstractMABSolver
import numpy as np
from scipy.stats import  gamma, uniform
import cvxpy

class HTINF(AbstractMABSolver):

    def __init__(self, N, alpha, sigma):
        super().__init__(N, 1 + alpha, sigma)
        self.theta_alpha = min(1 - 2 ** ((1 - self.alpha) / (2 * self.alpha - 1)), (2 - 2/self.alpha) ** (1/(2 - self.alpha)))
        self.weighted_losses = [np.zeros(self.N)]

    def psi(self, x):
        return -1 * self.alpha * sum(x ** 1/self.alpha)

    def argmin(self, eta, loss):
        P = cvxpy.Variable(self.N)
        obj = cvxpy.Minimize(eta * cvxpy.sum(loss @ P) - self.alpha * cvxpy.sum(P ** (1/self.alpha)))
        problem = cvxpy.Problem(obj, [sum(P) == 1, P >= 0])
        problem.solve()
        return P.value
    
    def choose_arm(self):
        self.lr = 1/(self.sigma * self.current_step ** (1 / self.alpha))
        probs = abs(self.argmin(self.lr, np.vstack(self.weighted_losses)))
        probs /= probs.sum()
        return np.random.choice(self.N, p=probs), probs
    
    def update_solver(self, loss, arm, probs):
        threshold_t = self.theta_alpha * probs[arm] ** (1 / self.alpha) / self.lr
        weighted_losses_update = np.zeros(self.N)
        if loss <= threshold_t:
            weighted_losses_update[arm] = loss / probs[arm]
        self.weighted_losses.append(weighted_losses_update)