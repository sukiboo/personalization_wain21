
import numpy as np
from bandit_environment import GymHyperpersonalizationEnv
from online_rl import OnlineRL
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import dill as pickle
import os
import time

np.set_printoptions(precision=4)
sns.set_theme(style='darkgrid', palette='muted')


class ExperimentDataGenerator:
    '''solve synthetic hyperpersonalization tasks via various methods'''

    def __init__(self, params_exp, params_env, params_rl):
        self.__dict__.update(params_exp)
        self.params_env = params_env
        self.params_rl = params_rl
        self.configure()

    def configure(self):
        '''configure parameters and fix random seeds for reproducibility'''
        np.random.seed(self.seed)
        self.exp_seeds = np.random.randint(1e+09, size=self.num_exp)
        self.seed_rl = np.random.randint(1e+09)
        self.data = {}
        self.params_rl.update({'learn_steps': self.timesteps, 'seed': self.seed_rl})

    def setup_environment(self, seed):
        '''setup the hyperpersonalization task with the given seed'''
        self.params_env.update({'seed': seed})
        self.env = GymHyperpersonalizationEnv(self.params_env)
        r_avg, r_min, r_max = self.env.get_r_stats(steps=self.timesteps)
        data = {'AVG': r_avg, 'MIN': r_min, 'MAX': r_max}
        return data

    def run_experiments(self, save=True, plot=True):
        '''compute and save/plot experiment results'''
        for exp in range(self.num_exp):
            print(f'running experiment {exp+1}/{self.num_exp}...')
            data_env = self.setup_environment(seed=self.exp_seeds[exp])
            data_rl = OnlineRL(self.env, self.params_rl).run_simulations()
            self.data[exp] = data_env
            self.data[exp].update(data_rl)
        if save:
            self.save_variables()
        self.process_exp_data()
        if plot:
            self.plot_exp_data()

    def process_exp_data(self):
        '''create the list of dataframes with experiment results'''
        self.exp_data = []
        for exp in range(self.num_exp):
            self.exp_data.append(pd.DataFrame(self.data[exp]))

    def plot_exp_data(self):
        '''plot results of the tests'''
        df = sum(self.exp_data) / len(self.exp_data)
        df = df.sub(df['AVG'], axis=0)
        df = df.div(df['MAX'], axis=0)
        df = df.drop(['AVG', 'MIN', 'MAX'], axis=1)
        df.plot(figsize=(8,5), linewidth=3, alpha=.75)
        plt.tight_layout()
        os.makedirs('./images/', exist_ok=True)
        plt.savefig(f'./images/tests_{time.strftime("%Y-%m-%d_%H.%M.%S")}.pdf', format='pdf')
        plt.show()

    def save_variables(self):
        '''save class variables to a file'''
        os.makedirs('./save/', exist_ok=True)
        save_name = f'exp_{time.strftime("%Y-%m-%d_%H.%M.%S")}.pkl'
        with open('./save/' + save_name, 'wb') as save_file:
            pickle.dump(self.__dict__, save_file)

    def load_variables(self, save_name):
        '''load class variables from a file'''
        try:
            with open('./save/' + save_name, 'rb') as save_file:
                self.__dict__.update(pickle.load(save_file))
            self.process_exp_data()
            self.plot_exp_data()
        except:
            raise NameError(f'\ncannot load file {save_name}...')


if __name__ == '__main__':
    '''experiment parameters'''
    params_exp = {'num_exp': 10, 'timesteps': 100000, 'seed': 0}
    '''environment parameters'''
    params_s = {'dim_layers': [10,10,10], 'weight_norm': 1.0}
    params_a = {'dim_layers': [10,10,10], 'weight_norm': 1.0}
    params_env = {'dim_s': 100, 'dim_a': 100, 'dim_feature': 10, 'num_a': 1000,
                  's_low': -1, 's_high': 1, 'a_low': -1, 'a_high': 1,
                  'params_s': params_s, 'params_a': params_a}
    '''rl parameters'''
    params_rl = {'algos': ['A2C', 'DQN', 'PPO'], 'num_sim': 3}
    '''run simulations'''
    exp = ExperimentDataGenerator(params_exp, params_env, params_rl)
    exp.run_experiments()
    # exp.load_variables('exps.pkl')

