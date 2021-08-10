
import numpy as np
from hyperpersonalization_gym_env import HyperpersonalizationGymEnv
from online_rl import OnlineRL
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import dill as pickle
import time

np.set_printoptions(precision=4)
sns.set_theme(style='darkgrid', palette='muted')


class TestDataGenerator:
    '''solve synthetic hyperpersonalization tasks via various methods'''

    def __init__(self, params, params_env, params_opt):
        self.__dict__.update(params)
        self.params_env = params_env
        self.params_opt = params_opt
        self.configure()

    def configure(self):
        '''generate lists of random seeds for reproducibility'''
        self.data = {}
        np.random.seed(self.seed)
        self.seed_list = np.random.randint(1e+09, size=self.num_tests)
        self.sim_seeds = np.random.randint(1e+09, size=(self.num_tests,self.num_sims))

    def setup_environment(self, seed):
        '''setup the hyperpersonalization task'''
        self.params_env.update({'seed': seed})
        self.env = HyperpersonalizationGymEnv(self.params_env)

    def get_reward_vals(self, seeds):
        '''compute the average/minimial/maximal reward values'''
        data = {}
        for i in range(self.num_sims):
            data[i] = {'AVG': {}, 'MIN': {}, 'MAX': {}}
            self.env.set_seed(seed=seeds[i])
            for j in range(self.timesteps):
                s = self.env.reset()
                r_avg, r_min, r_max = self.env.env.get_r_vals(s)
                data[i]['AVG'][j+1] = r_avg.item()
                data[i]['MIN'][j+1] = r_min.item()
                data[i]['MAX'][j+1] = r_max.item()
        return data

    def run_tests(self, save=True, plot=True):
        '''compute and save/plot test results'''
        self.params_opt.update({'learn_steps': self.timesteps})
        for test in range(self.num_tests):
            print(f'running test {test+1}/{self.num_tests}...')
            self.setup_environment(seed=self.seed_list[test])
            self.data[test] = self.get_reward_vals(seeds=self.sim_seeds[test])
            online_rl = OnlineRL(self.env, self.params_opt)
            data_rl = online_rl.run_simulations(seeds=self.sim_seeds[test])
            for sim in range(self.num_sims):
                self.data[test][sim].update(data_rl[sim])
        if save:
            self.save_variables()
        self.process_test_data()
        if plot:
            self.plot_test_data()

    def process_test_data(self):
        '''create the list of dataframes with test results'''
        self.test_data = []
        for test in range(self.num_tests):
            for sim in range(self.num_sims):
                self.test_data.append(pd.DataFrame(self.data[test][sim]))

    def plot_test_data(self):
        '''plot results of the tests'''
        df = sum(self.test_data) / len(self.test_data)
        df = df.sub(df['AVG'], axis=0)
        df = df.div(df['MAX'], axis=0)
        df = df.drop(['AVG', 'MIN', 'MAX'], axis=1)
        df.plot(figsize=(8,5), linewidth=3, alpha=.75)
        plt.tight_layout()
        plt.savefig('./tests.pdf', format='pdf')
        plt.show()

    def save_variables(self):
        '''save class variables to a file'''
        os.makedirs('./save/', exist_ok=True)
        save_name = f'tests_{time.strftime("%Y-%m-%d_%H.%M.%S")}.pkl'
        with open('./save/' + save_name, 'wb') as save_file:
            pickle.dump(self.__dict__, save_file)

    def load_variables(self, save_name):
        '''load class variables from a file'''
        try:
            with open('./save/' + save_name, 'rb') as save_file:
                self.__dict__.update(pickle.load(save_file))
            self.process_test_data()
            self.plot_test_data()
        except:
            raise NameError(f'\ncannot load file {save_name}...')


if __name__ == '__main__':

    '''simulation parameters'''
    params = {'num_tests': 10, 'num_sims': 10, 'timesteps': 2, 'seed': 0}

    '''environment parameters'''
    params_s = {'dim_layers': [10,10,10], 'weight_norm': 1.0}
    params_a = {'dim_layers': [10,10,10], 'weight_norm': 1.0}
    params_env = {'dim_s': 100, 'dim_a': 100, 'dim_feature': 10, 'num_a': 1000,
                  's_low': -5, 's_high': 5, 'r_low': -5, 'r_high': 5,
                  'params_s': params_s, 'params_a': params_a}

    '''online rl parameters'''
    params_opt = {'algos': ['A2C', 'DQN', 'PPO']}

    '''run simulations'''
    tdg = TestDataGenerator(params, params_env, params_opt)
    tdg.run_tests()

