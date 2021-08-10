# Synthetic Hyperpersonalization Environment

A *very* drafty version of the code on the Synthetic Hyperpersonalization Tasks (will rename it later).

Configure the experiments in `run_tests.py` and run with `python run_tests.py`.

## A brief overview of files
* `synthetic_gaussian_mapping.py` --- creates the Synthetic Gaussian Mapping that acts as a latent feature extractor for the simulated reward signal.
* `contextual_bandit_hyperpersonalization.py` --- creates the Synthetic Hyperpersonalization Environment with the simulated reward signal.
* `hyperpersonalization_gym_env.py` --- creates an OpenAI Gym environment based on the Synthetic Hyperpersonalization Environment created above (convenient for deploying RL algorithms).
* `online_rl.py` --- trains online RL algorithms on a given environment.
* `run_tests.py` --- sets up and runs the experiments.

There are some comments and docstrings in the code but I'm not sure how helpfulthey actually are. If all this seems like a horrible mess (which it totally is!), just let me know and I'll try to explain what's happening here.
