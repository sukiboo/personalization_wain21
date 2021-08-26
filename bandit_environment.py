
import numpy as np
import gym
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr
from synthetic_gaussian_mapping import SyntheticGaussianMapping

np.set_printoptions(precision=4)
gym.logger.set_level(40)
sns.set_theme(style='darkgrid', palette='muted', font='monospace')


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
        np.random.seed(seed)
        self.rng_s = np.random.default_rng(seed=self.seed)
        self.rng_a = np.random.default_rng(seed=self.seed+1)
        self.rng_r = np.random.default_rng(seed=self.seed+2)

    def get_state(self, num_s=1):
        '''generate observed states'''
        S = self.rng_s.uniform(self.s_low, self.s_high, (num_s,self.dim_s))
        return S

    def sample_actions(self):
        '''generate the set of available actions'''
        self.A = self.rng_a.uniform(self.a_low, self.a_high, (self.num_a,self.dim_a))

    def create_state_embedding(self):
        '''generate state feature map'''
        self.params_s = {'dim_in': self.dim_s, 'dim_layers': self.r_arch,
                         'dim_out': self.dim_feature, 'seed': self.rng_r.integers(1e+09)}
        self.feature_map_s = SyntheticGaussianMapping(self.params_s)
        self.feature_s = lambda s: self.feature_map_s.propagate(s)

    def create_action_embedding(self):
        '''generate action feature map'''
        self.params_a = {'dim_in': self.dim_a, 'dim_layers': self.r_arch,
                         'dim_out': self.dim_feature, 'seed': self.rng_r.integers(1e+09)}
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

    def get_r_vals(self, s):
        '''compute the average, minimum, and maximum reward values for a given state'''
        r_vals = self.r(s, self.A)
        r_avg = r_vals.mean(axis=1)
        r_min = r_vals.min(axis=1)
        r_max = r_vals.max(axis=1)
        return r_avg, r_min, r_max

    def visualize_reward_distribution(self, num_s=5, num_a=10):
        '''visualize reward distribution across the available actions'''
        S = np.random.randn(num_s, self.dim_s) / 100
        R = pd.DataFrame(self.r(S, self.A[:num_a]).T)
        ax = R.plot.bar(figsize=(8,5), width=.8, rot=0, legend=None)
        ax.set_xlabel('available actions')
        ax.set_ylabel('reward')
        os.makedirs('./images/', exist_ok=True)
        plt.savefig('./images/reward_distribution.pdf', format='pdf')
        plt.show()

    def visualize_reward_correlation(self, num_s=100000):
        '''visualize reward correlations across different clusters'''
        S = self.get_state(num_s=num_s)
        R = self.r(S, self.A)
        clusters = KMeans(n_clusters=self.num_a).fit(S).labels_
        corr = np.zeros(shape=(2,self.num_a))
        for k in range(self.num_a):
            k_ind = np.where(clusters==k)[0]
            corr[:,k] = pearsonr(pdist(S[k_ind]), pdist(R[k_ind]))
        fig, ax = plt.subplots(figsize=(8,5))
        ax.bar(np.arange(1,self.num_a+1), corr[0], width=1.)
        ax.set_title(f'Pearson correlation (p = {np.mean(corr[1]):.2e})')
        ax.set_xlabel('state space clusters')
        os.makedirs('./images/', exist_ok=True)
        plt.savefig('./images/reward_correlation.pdf', format='pdf')
        plt.show()


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


class GymStateClusteringEnv(GymHyperpersonalizationEnv):
    '''create custom gym environment with a clustered state space'''

    def __init__(self, params_env, params_cluster):
        GymHyperpersonalizationEnv.__init__(self, params_env)
        self.__dict__.update(params_cluster)
        self.action_space = gym.spaces.Discrete(self.env.num_a)
        self.observation_space = gym.spaces.Discrete(self.K)
        self.cluster_state_space()

    def cluster_state_space(self):
        '''cluster the state space'''
        S = self.env.get_state(self.timesteps)
        self.clustering = KMeans(n_clusters=self.K).fit(S)
        self.reset_env()

    def step(self, action_index):
        '''given an observed state take an action and receive reward'''
        self.action = self.env.A[action_index]
        self.reward = self.env.r(self.state_vector, self.action).item()
        done = True
        info = {}
        return self.state, self.reward, done, info

    def reset(self):
        '''observe a new state'''
        self.state_vector = self.env.get_state()
        self.state = self.clustering.predict(self.state_vector).item()
        return self.state


if __name__ == '__main__':
    '''configure the environment'''
    params_env = {'num_a': 100, 'dim_s': 100, 'dim_a': 100, 'dim_feature': 100,
                  's_low': -1, 's_high': 1, 'a_low': -1, 'a_high': 1,
                  'r_arch': [100,100,100], 'seed': 2021}
    '''create the environment'''
    env = HyperpersonalizationEnv(params_env)
    env.visualize_reward_distribution()
    env.visualize_reward_correlation()

