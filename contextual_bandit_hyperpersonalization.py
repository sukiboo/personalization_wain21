
from synthetic_gaussian_mapping import SyntheticGaussianMapping
import numpy as np
np.set_printoptions(precision=4)


class ContextualBanditHyperpersonalization:
    '''generate synthetic contextual bandit for a simple hyperpersonalization task'''

    def __init__(self, params_env):
        self.__dict__.update(params_env)
        self.set_random_seed()
        self.sample_actions()
        self.create_state_features()
        self.create_action_features()
        self.create_reward_function()

    def set_random_seed(self):
        '''fix random seed for reproducibility'''
        np.random.seed(self.seed)
        self.seed_s = np.random.randint(1e+09)
        self.seed_a = np.random.randint(1e+09)

    def get_state(self, num_s=1):
        '''generate observed state'''
        self.S = np.clip(np.random.randn(num_s, self.dim_s), self.s_low, self.s_high)
        return self.S

    def sample_actions(self):
        '''generate the set of available actions'''
        self.A = np.random.randn(self.num_a, self.dim_a)

    def create_state_features(self):
        '''generate state feature map'''
        self.params_s.update({'dim_in': self.dim_s,
                              'dim_out': self.dim_feature,
                              'seed': self.seed_s})
        self.feature_map_s = SyntheticGaussianMapping(self.params_s)
        self.feature_s = lambda s: self.feature_map_s.propagate(s)

    def create_action_features(self):
        '''generate action feature map'''
        self.params_a.update({'dim_in': self.dim_a,
                              'dim_out': self.dim_feature,
                              'seed': self.seed_a})
        self.feature_map_a = SyntheticGaussianMapping(self.params_a)
        self.feature_a = lambda a: self.feature_map_a.propagate(a)

    def create_reward_function(self):
        '''reward function of the environment'''
        feature_relevance = lambda s,a: self.relevance(self.feature_s(s), self.feature_a(a))
        self.r = lambda s,a: np.clip(feature_relevance(s,a), self.r_low, self.r_high)
        self.r_avg = lambda s: self.r(s, self.A).mean(axis=1)
        self.r_min = lambda s: self.r(s, self.A).min(axis=1)
        self.r_max = lambda s: self.r(s, self.A).max(axis=1)

    def relevance(self, s, a):
        '''relevance function in the latent feature space'''
        norm_s = np.linalg.norm(s, axis=1, keepdims=True)
        norm_a = np.linalg.norm(a, axis=1, keepdims=True)
        cosine_sim = np.matmul(s, a.T) / np.matmul(norm_s, norm_a.T)
        return np.tan(cosine_sim * np.pi / 2)

    def get_r_vals(self, s):
        '''compute average, minimum, and maximum reward values for a given state'''
        r_vals = self.r(s, self.A)
        r_avg = r_vals.mean(axis=1)
        r_min = r_vals.min(axis=1)
        r_max = r_vals.max(axis=1)
        return r_avg, r_min, r_max


if __name__ == '__main__':
    '''configure the environment'''
    params_s = {'dim_layers': [10,10], 'weight_norm': 1.0}
    params_a = {'dim_layers': [10,10], 'weight_norm': 1.0}
    params_env = {'dim_s': 10, 'dim_a': 10, 'dim_feature': 10, 'num_a': 100,
                  's_low': -5, 's_high': 5, 'r_low': -5, 'r_high': 5,
                  'params_s': params_s, 'params_a': params_a, 'seed': 0}
    '''create the environment'''
    env = ContextualBanditHyperpersonalization(params_env)

