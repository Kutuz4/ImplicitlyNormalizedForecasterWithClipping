# ImplicitlyNormalizedForecasterWithClipping

In this repository we provide code for experiments with different popular MultiArmBandits solvers and heavy-tailed noise in reward functions. We also demonstrate our novel solver from paper  
https://arxiv.org/abs/2305.06743.

In order to run code:

- clone this repo and install requirements 


- type command

`python3 main.py --n_arms 2 --iterations 8000 --n_runs 100 --plot_num 2 --output output_folder_name `

where

- `n_arms ` is number of arms for each bandit
- `iterations` is number of iterations for each algorithm per one run
- `n_runs` is number of runs for each algorithm
- `plot_num` is number of plots that will be saved during experiment
- `output` is name of output folder for plots
