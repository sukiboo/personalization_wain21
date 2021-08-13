
from synthetic_gaussian_mapping import SyntheticGaussianMapping
import numpy as np
import gym

np.set_printoptions(precision=4)
gym.logger.set_level(40)


class HyperpersonalizationEnv:
    '''generate contextual bandit for a synthetic hyperpersonalization task'''

    def __init__(self, params_env):
        self.__dict__.update(params_env)
        self.set_random_seed()
        self.sample_actions()
        self.create_reward_function()

    def set_random_seed(self, seed=None):
        '''fix random seed for reproducibility'''
        if seed is not None:
            self.seed = seed
        self.rng_s = np.random.default_rng(seed=self.seed)
        self.rng_a = np.random.default_rng(seed=self.seed+1)
        self.rng_r = np.random.default_rng(seed=self.seed+2)
        # np.random.seed(self.seed)
        # self.seed_s = np.random.randint(1e+09)
        # self.seed_a = np.random.randint(1e+09)

    def get_state(self, num_s=1):
        '''generate observed states'''
        # self.S = np.clip(self.rng.standard_normal((num_s,self.dim_s)), self.s_low, self.s_high)
        ##self.S = np.clip(self.rng.standard_normal((num_s,self.dim_s)), self.s_low, self.s_high)
        self.S = self.rng_s.uniform(self.s_low, self.s_high, (num_s,self.dim_s))
        return self.S

    def sample_actions(self):
        '''generate the set of available actions'''
        # self.A = np.random.randn(self.num_a, self.dim_a)
        ##self.A = self.rng.standard_normal((self.num_a,self.dim_a))
        self.A = self.rng_a.uniform(self.a_low, self.a_high, (self.num_a,self.dim_a))

    def create_state_embedding(self):
        '''generate state feature map'''
        self.params_s.update({'dim_in': self.dim_s,
                              'dim_out': self.dim_feature,
                              'seed': self.rng_r.integers(1e+09)})
        self.feature_map_s = SyntheticGaussianMapping(self.params_s)
        self.feature_s = lambda s: self.feature_map_s.propagate(s)

    def create_action_embedding(self):
        '''generate action feature map'''
        self.params_a.update({'dim_in': self.dim_a,
                              'dim_out': self.dim_feature,
                              'seed': self.rng_r.integers(1e+09)})
        self.feature_map_a = SyntheticGaussianMapping(self.params_a)
        self.feature_a = lambda a: self.feature_map_a.propagate(a)

    def feature_relevance(self, s, a):
        '''measure feature relevance in the latent feature space'''
        norm_s = np.linalg.norm(s, axis=1, keepdims=True)
        norm_a = np.linalg.norm(a, axis=1, keepdims=True)
        cosine_sim = np.matmul(s, a.T) / np.matmul(norm_s, norm_a.T)
        return cosine_sim

    def create_reward_function(self):
        '''reward function of the environment'''
        self.create_state_embedding()
        self.create_action_embedding()
        self.r = lambda s,a: self.feature_relevance(self.feature_s(s), self.feature_a(a))
        # self.r_avg = lambda s: self.r(s, self.A).mean(axis=1)
        # self.r_min = lambda s: self.r(s, self.A).min(axis=1)
        # self.r_max = lambda s: self.r(s, self.A).max(axis=1)

    def get_r_vals(self, s):
        '''compute the average, minimum, and maximum reward values for a given state'''
        r_vals = self.r(s, self.A)
        r_avg = r_vals.mean(axis=1)
        r_min = r_vals.min(axis=1)
        r_max = r_vals.max(axis=1)
        return r_avg, r_min, r_max


class GymHyperpersonalizationEnv(gym.Env):
    '''create custom gym environment for a hyperpersonalization task'''

    def __init__(self, params_env):
        super(GymHyperpersonalizationEnv, self).__init__()
        self.env = HyperpersonalizationEnv(params_env)
        self.action_space = gym.spaces.Discrete(self.env.num_a)
        self.observation_space = gym.spaces.Box(low=self.env.s_low, high=self.env.s_high,
                                                shape=(self.env.dim_s,), dtype=np.float)

    def reset_env(self):
        '''re-seed the environment'''
        self.env.set_random_seed()

    def step(self, action_index):
        '''given an observed state take an action and receive reward'''
        self.action = self.env.A[action_index]
        self.reward = self.env.r(self.state, self.action).item()
        done = True
        info = {}
        return self.state, self.reward, done, info

    def reset(self):
        '''observe a new state'''
        self.state = self.env.get_state().flatten()
        return self.state

    def get_r_stats(self, steps):
        '''compute the average, minimum, and maximum reward values'''
        S = self.env.get_state(steps)
        stats = self.env.get_r_vals(S)
        self.reset_env()
        return stats


if __name__ == '__main__':
    '''configure the environment'''
    params_s = {'dim_layers': [10,10], 'weight_norm': 1.0}
    params_a = {'dim_layers': [10,10], 'weight_norm': 1.0}
    params_env = {'dim_s': 10, 'dim_a': 10, 'dim_feature': 10, 'num_a': 1000,
                  's_low': -1, 's_high': 1, 'a_low': -1, 'a_high': 1,
                  'params_s': params_s, 'params_a': params_a, 'seed': 0}
    '''create the environment'''
    env = GymHyperpersonalizationEnv(params_env)
