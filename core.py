import argparse
from bandits.APEBandits import APEBandits 
from bandits.OMDTsalisEntropyBandits import OMDTsalisEntropyBandits
from bandits.HTINF import HTINF
from bandits.UCBpsiMeanTrunc import UCBpsiMeanTrunc
from tools import *
from plotting import *
import os
from tqdm import tqdm
from copy import deepcopy

def parse_args():
    parser = argparse.ArgumentParser(description='Bandit Comparison')
    parser.add_argument(
        '--iterations',
        type=int,
        default=1000,
        help='provide num of iterations for each algorithm (default: 1000)',
        dest='iterations'
    )
    parser.add_argument(
        '--n_arms',
        type=int,
        default=2,
        help='provide num of arms for bandits (default: 2)',
        dest='n_arms'
    )
    parser.add_argument(
        '--n_runs',
        type=int,
        default=10,
        help='provide num of runs for each algo(default: 10)',
        dest='n_runs'
    )
    parser.add_argument(
        '--plot_num',
        type=int,
        default=1,
        help='provide num of plots(default: 1)',
        dest='plot_num'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='provide random seed(default: 42)',
        dest='seed'
    )

    parser.add_argument('--output',
                         default = '',
                         help='select output file or directory', 
                         dest='output_path')

    args = parser.parse_args()
    return args

def run():
    args = parse_args()

    T = args.iterations
    K = args.n_runs
    n = args.n_arms
    seed = args.seed
    plot_num = args.plot_num
    plot_step = (K-1)//plot_num
    dirpath = args.output_path


    alphas = [0.025, 0.05, 0.1]
    scales = [3, 3.1]
    stats = {"UCB": {}, "INFC": {}, "HTINF": {}, "APE": {}}
    to_list = lambda x: x if isinstance(x, list) else x.tolist()

    logs_template = {"regrets": [], "losses": [], 
                     "expected_regrets": [], "arms_seq": [],
                     "best_arm_probabilities": []}


    

    for alpha in alphas:
        np.random.seed(seed)

   
        distrs, means, best_arm, eps = create_scaled_distributions(n, alpha, scales)
        sigma = max(d.alpha_moment for d in distrs)
        solvers = {"INFC": (OMDTsalisEntropyBandits, (n, alpha, sigma, T)),
                    "HTINF": (HTINF, (n, alpha, sigma)),
                      "UCB": (UCBpsiMeanTrunc, (n, alpha, sigma, 2, eps)),
                        "APE": (APEBandits, (n, alpha, sigma, T // 10))}
    
        for key in solvers:
            stats[key][alpha] = deepcopy(logs_template)

        for k in tqdm(range(K)):

            curr_solvers = {solver_name: solver(*args) for solver_name, (solver, args) in solvers.items()}

            for _ in range(T):
                losses_arms = [x.rvs() for x in distrs]
                for solver in curr_solvers.values():
                    solver(losses_arms, means)

            for solver_name, solver in curr_solvers.items():
                solver_logs = solver.log()
                for stat_name, logs in solver_logs.items():
                    stats[solver_name][alpha][stat_name].append(to_list(logs).copy())
        
            if k % plot_step == 0 and k > 0:
                name = os.path.join(dirpath, f"{k}_tracks_a_{alpha}_n_{n}_delta_{eps}.pdf")
                plot_solvers_stats(name, stats, f"Statistics for {k} tracks, alpha={alpha}, arms count {n} and delta={eps}", alpha, w=100)