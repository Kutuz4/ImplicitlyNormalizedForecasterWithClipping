from tools import *
import matplotlib.pyplot as plt

def mean_conf_plotting(data, ax, label, color="r", std_mul=1, cut_step=5, w=3):
    mean = moving_average(data.mean(axis=1)[cut_step:], w=w)
    std = data.std(axis=1)[cut_step + w - 1:]
    x = np.arange(1, len(data) + 1, 1)[cut_step + w - 1:]
    ax.plot(x, mean, color=color, label=label)
    ax.fill_between(x, mean - std_mul * std, mean + std_mul * std, color=color, alpha=0.1)


def plot_solvers_stats(path, stats, title, alpha=0.1, colors="rgbc", w=3):
        fig, axes = plt.subplots(ncols=2, figsize=(16, 6))
            
        fig.suptitle(title)
            
        legend = []
        for i, solver_name in enumerate(stats.keys()):
            axes[0].plot(np.mean(np.array(stats[solver_name][alpha]["expected_regrets"]), axis=0)[1:], color=colors[i])
            best_probs = np.array(stats[solver_name][alpha]["best_arm_probabilities"])
            mean_conf_plotting(best_probs.T, ax=axes[1], label=solver_name, cut_step=0, color=colors[i], w=w)
            legend.append(solver_name)

        axes[0].set_title("Expected regret")
        axes[1].set_title("Probability of best arm choice")
        axes[0].legend(legend)
            
        for i in range(2):
            axes[i].grid(True)
        plt.savefig(path)