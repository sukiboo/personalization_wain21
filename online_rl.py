
import numpy as np
import gym
import torch
import stable_baselines3 as sb3
from stable_baselines3.common.callbacks import BaseCallback

np.set_printoptions(precision=4)
gym.logger.set_level(40)


class Callback(BaseCallback):
    '''custom callback that records rewards on each timestep'''

    def __init__(self, data, max_timestep):
        super(Callback, self).__init__()
        self.data = data
        self.max_timestep = max_timestep

    def _on_step(self):
        t = self.num_timesteps
        if t <= self.max_timestep:
            r = self.locals['infos'][0]['episode']['r']
            self.data[t] = r
        return True


class OnlineRL:
    '''train on-policy agents'''

    def __init__(self, env, params):
        self.__dict__.update(params)
        self.env = env
        self.data = {}

    def set_random_seed(self, seed):
        '''fix random seed for reproducibility'''
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.env.set_seed(seed)

    def train_agent(self, alg, seed=0, i=0):
        '''train an agent with an rl algorithm'''
        self.data[i][alg] = {}
        self.set_random_seed(seed)
        if alg == 'A2C':
            agent = sb3.A2C('MlpPolicy', self.env, verbose=0)
        elif alg == 'DQN':
            agent = sb3.DQN('MlpPolicy', self.env, verbose=0)
        elif alg == 'PPO':
            agent = sb3.PPO('MlpPolicy', self.env, verbose=0)
        else:
            self.data[i].pop(alg, None)
            raise NameError(f'\nalgorithm {alg} is not implemented...')
        print(f'    training {alg}-agent...')
        agent.learn(self.learn_steps,
                    callback=Callback(data=self.data[i][alg],
                    max_timestep=self.learn_steps))

    def run_simulations(self, seeds):
        '''train agents on simulated environment'''
        for i in range(len(seeds)):
            self.data[i] = {}
            print(f'  running simulation {i+1}/{len(seeds)}:')
            for alg in self.algos:
                self.train_agent(alg, seeds[i], i)
        return self.data

