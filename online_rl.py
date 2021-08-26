
import numpy as np
import gym
import torch
import stable_baselines3 as sb3

np.set_printoptions(precision=4)


class Callback(sb3.common.callbacks.BaseCallback):
    '''custom callback that records reward on each timestep'''

    def __init__(self, data, max_timestep):
        super(Callback, self).__init__()
        self.data = data
        self.max_timestep = max_timestep

    def _on_step(self):
        t = self.num_timesteps
        if t <= self.max_timestep:
            r = self.locals['infos'][0]['episode']['r']
            self.data.append(r)
        return True


class OnlineRL:
    '''train online rl agents'''

    def __init__(self, env, params_rl):
        self.__dict__.update(params_rl)
        self.env = env
        self.data = {}
        self.policy_kwargs = {'net_arch': self.net_arch}

    def set_random_seed(self, seed):
        '''fix random seed for reproducibility'''
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.env.reset_env()

    def run_simulations(self):
        '''train agents on simulated environment'''
        np.random.seed(self.seed)
        seed_list = np.random.randint(1e+09, size=self.num_sim)
        for alg in self.algos:
            print(f'  training {alg}-agent...')
            data_alg = []
            for seed in seed_list:
                self.set_random_seed(seed)
                data_alg.append(self.train_agent(alg))
            self.data[alg] = np.mean(data_alg, axis=0)
        return self.data

    def train_agent(self, alg):
        '''train an rl agent with a specified algorithm'''
        if alg == 'A2C':
            agent = sb3.A2C('MlpPolicy', self.env, verbose=0, policy_kwargs=self.policy_kwargs)
        elif alg == 'DQN':
            agent = sb3.DQN('MlpPolicy', self.env, verbose=0, policy_kwargs=self.policy_kwargs)
        elif alg == 'PPO':
            agent = sb3.PPO('MlpPolicy', self.env, verbose=0, policy_kwargs=self.policy_kwargs)
        else:
            raise NameError(f'\nalgorithm {alg} is not implemented...')
        data = []
        agent.learn(self.learn_steps,
                    callback=Callback(data=data, max_timestep=self.learn_steps))
        return data

