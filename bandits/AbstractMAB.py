import numpy as np

class AbstractMABSolver:

    def __init__(self, N, alpha, sigma, **kwargs):
        self.N = N
        self.alpha = alpha
        self.sigma = sigma
        self.arms_seq = [].copy()
        self.regrets = [0].copy()
        self.expected_regrets = [0].copy()
        self.losses = [].copy()
        self.best_arm_probabilities = [].copy()
        self.current_step = 0
        self.kwargs = kwargs

    def choose_arm(self):
        pass

    def update_solver(self, loss, arm, probs):
        pass

    def __call__(self, arms_losses, arm_expectations):
        self.current_step += 1
        best_arm = np.argmin(arm_expectations)
        arm, probabilities = self.choose_arm()
        loss = arms_losses[arm]
        self.update_solver(loss, arm, probabilities)
        self.regrets.append(self.regrets[-1] + arms_losses[arm] - arms_losses[best_arm])
        self.arms_seq.append(arm)
        self.expected_regrets.append(self.expected_regrets[-1] + arm_expectations[arm] - arm_expectations[best_arm])
        self.losses.append(arms_losses[arm])
        self.best_arm_probabilities.append(probabilities[best_arm])

    def log(self):
        norm_ = lambda lst: np.array(lst[1:]) / np.array([i + 1 for i in range(self.current_step)])
        return {"regrets": norm_(self.regrets), "losses": self.losses,
                 "expected_regrets": norm_(self.expected_regrets), "arms_seq": self.arms_seq,
                   "best_arm_probabilities": self.best_arm_probabilities}
    