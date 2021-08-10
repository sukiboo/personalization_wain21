
from contextual_bandit_hyperpersonalization import ContextualBanditHyperpersonalization
import numpy as np
import gym
from stable_baselines3.common.env_checker import check_env

np.set_printoptions(precision=4)
gym.logger.set_level(40)


class HyperpersonalizationGymEnv(gym.Env):
    '''create custom gym environment for a hyperpersonalization task'''

    def __init__(self, params):
        super(HyperpersonalizationGymEnv, self).__init__()
        self.env = ContextualBanditHyperpersonalization(params)
        self.action_space = gym.spaces.Discrete(self.env.num_a)
        self.observation_space = gym.spaces.Box(low=self.env.s_low, high=self.env.s_high,
                                                shape=(self.env.dim_s,), dtype=np.float)
        check_env(self, warn=True)

    def step(self, action_index):
        self.action = self.env.A[action_index]
        self.reward = self.env.r(self.state, self.action).item()
        done = True
        info = {}
        return self.state, self.reward, done, info

    def set_seed(self, seed):
        np.random.seed(seed)
        self.seed(seed)
        self.action_space.seed(int(seed))

    def reset(self):
        self.state = self.env.get_state().flatten()
        return self.state


if __name__ == '__main__':
    '''configure the environment'''
    params_s = {'dim_layers': [10,10], 'weight_norm': 1}
    params_a = {'dim_layers': [10,10], 'weight_norm': 1}
    params_env = {'dim_s': 10, 'dim_a': 10, 'dim_feature': 10, 'num_a': 100,
                  's_low': -5, 's_high': 5, 'r_low': -5, 'r_high': 5,
                  'params_s': params_s, 'params_a': params_a, 'seed': 0}
    '''create the environment'''
    env = HyperpersonalizationGymEnv(params_env)

