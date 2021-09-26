# Synthetic Personalization Environment

This repository contains the source code for the numerical experiments presented in the paper [On the Unreasonable Efficiency of State Space Clustering in Personalization Tasks](https://arxiv.org/).

## Installation
* Install the requirements via `pip install -r requirements.txt`
* Configure the environment in `config.yaml`
* Run the experiments via `python -m run_experiments`

## Files Overview
* `synthetic_gaussian_mapping.py` --- creates the Synthetic Gaussian Mapping that acts as a latent feature extractor for the simulated reward signal
* `bandit_environment.py` --- creates the Synthetic Hyperpersonalization Environment with the simulated reward signal as an OpenAI Gym environment
* `online_rl.py` --- trains online RL algorithms on a given environment
* `run_experiments.py` --- sets up and runs the experiments
* `config.yaml` --- stores the environment/training/experiment parameters
* `requirements.txt` --- lists the required packages

## License
This project is licensed under the MIT License.
