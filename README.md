# Synthetic Hyperpersonalization Environment

A *very* drafty version of the code on the Synthetic Hyperpersonalization Tasks (will rename it later).

Configure the experiments in `run_experiments.py` and run with `python run_experiments.py`.

## A brief overview of files
* `synthetic_gaussian_mapping.py` --- creates the Synthetic Gaussian Mapping that acts as a latent feature extractor for the simulated reward signal.
* `bandit_environment.py` --- creates the Synthetic Hyperpersonalization Environment with the simulated reward signal as an OpenAI Gym environment (convenient for deploying RL algorithms).
* `online_rl.py` --- trains online RL algorithms on a given environment.
* `run_experiments.py` --- sets up and runs the experiments.

There are some comments and docstrings in the code but I'm not sure how helpfulthey actually are. If all this seems like a horrible mess (which it totally is!), just let me know and I'll try to explain what's happening here.
