from scipy.stats import rv_continuous
from scipy.integrate import quad
from scipy.optimize import fsolve
import numpy as np

def compute_f(x, alpha):
    return 1/(x ** (2 + alpha) * np.log(x) ** 2) if x >= 2 else 0


def create_degree_log_q_distr(alpha, scale):
    f = lambda x: compute_f(x / scale, alpha) / scale
    norm = quad(f, 2 * scale, np.inf, limit=500)[0]
    print(quad(lambda x: f(x) / norm, 2 * scale, np.inf, limit=500)[0])
    class DegreeLogQDistr(rv_continuous):

        def _pdf(self, x):
            return f(x)/norm

    tmp = DegreeLogQDistr(a=2 * scale)
    tmp.alpha_moment = quad(lambda x: (x) ** (1 + alpha) * f(x) / norm, 2 * scale, np.inf, limit=300)[0]
    tmp.alpha_moment = tmp.alpha_moment ** (1/(1 + alpha))
    tmp.mean_stat = quad(lambda x: (x) * f(x) / norm, 2 * scale, np.inf, limit=500)[0]
    return tmp

def create_scaled_distributions(n, alpha, scales=[3, 3.5]):
    distrs = [create_degree_log_q_distr(alpha, 1) for i in range(n)]
    means = [x.mean_stat for x in distrs]
    scales_ = [scales[i] / means[i] for i in range(n)]
    distrs = [create_degree_log_q_distr(alpha, scales_[i]) for i in range(n)]
    means = [x.mean_stat for x in distrs]
    min_arm = np.argmin(means)
    eps = np.round(abs(np.diff(means)[0]), 2)
    return distrs, means, min_arm, eps

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w




class SumRootsSolver:

    def __init__(self, N, start_beta=0.5, eps=1e-15):
        self.N = N
        self.start_beta = start_beta
        self.eps=1e-4

    def get_f(self, x: float, coefs):
        return sum(1/(x + coefs) ** 2) - 1

    def get_df(self, x: float, coefs): 
        return -2 * sum(1/(x + coefs) ** 3)

    def solve(self, coefs, return_steps=False):
        beta = self.start_beta
        return fsolve(lambda x: self.get_f(x, coefs), 0)