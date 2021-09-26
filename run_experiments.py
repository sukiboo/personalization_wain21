
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import dill as pickle
import os
import time
import yaml
from bandit_environment import GymHyperpersonalizationEnv, GymStateClusteringEnv
from online_rl import OnlineRL

np.set_printoptions(precision=4)
sns.set_theme(style='darkgrid', palette='tab20', font='monospace')


class ExperimentDataGenerator:
    '''solve synthetic hyperpersonalization tasks via various methods'''

    def __init__(self, params_exp, params_env, params_rl):
        self.__dict__.update(params_exp)
        self.params_env = params_env
        self.params_rl = params_rl
        self.params_cluster = {'K': params_env['num_a'], 'timesteps': self.timesteps}
        self.configure()

    def configure(self):
        '''configure parameters and fix random seeds for reproducibility'''
        np.random.seed(self.seed)
        self.exp_seeds = np.random.randint(1e+09, size=self.num_exp)
        self.seed_rl = np.random.randint(1e+09)
        self.data = {}
        self.params_rl.update({'learn_steps': self.timesteps, 'seed': self.seed_rl})
        self.save_name = f'exp_{self.params_env["num_a"]}_{self.params_env["dim_s"]}_'\
                         + f'{self.params_env["dim_a"]}_{self.params_env["dim_feature"]}_'\
                         + '-'.join(str(n) for n in self.params_env['r_arch']) + '_'\
                         + '-'.join(str(n) for n in self.params_rl['net_arch'])

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
            # get environment stats
            data_env = self.setup_environment(seed=self.exp_seeds[exp])
            self.data[exp] = data_env
            # rl on full environment
            data_rl = OnlineRL(self.env, self.params_rl).run_simulations()
            self.data[exp].update(data_rl)
            # rl on clustered environment
            self.env_cl = GymStateClusteringEnv(self.params_env, self.params_cluster)
            data_rl_cl = OnlineRL(self.env_cl, self.params_rl).run_simulations()
            data_rl_cl = {k + ' + K-means': v for k,v in data_rl_cl.items()}
            self.data[exp].update(data_rl_cl)
        if save:
            self.save_variables()
        self.process_exp_data()
        if plot:
            self.plot_exp_data()

    def process_exp_data(self):
        '''create the list of dataframes with normalized experiment results'''
        self.exp_data = []
        for exp in range(self.num_exp):
            df = pd.DataFrame(self.data[exp])
            df = df.sub(df['AVG'], axis=0)
            df = df.div(df['MAX'], axis=0)
            df = df.drop(['AVG', 'MIN', 'MAX'], axis=1)
            self.exp_data.append(df)

    def plot_exp_data(self):
        '''plot normalized results of the experiments'''
        df =  sum(self.exp_data) / self.num_exp
        df = df.rolling(100).mean()
        df = df.sort_index(axis=1)
        df.plot(figsize=(8,4.2), linewidth=3, alpha=.75)
        plt.legend(loc='lower right')
        plt.tight_layout()
        os.makedirs('./images/', exist_ok=True)
        plt.savefig('./images/' + self.save_name + '.pdf', format='pdf')
        plt.show()

    def save_variables(self):
        '''save class variables to a file'''
        os.makedirs('./save/', exist_ok=True)
        with open('./save/' + self.save_name + '.pkl', 'wb') as save_file:
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
    """
    '''experiment parameters'''
    params_exp = {'num_exp': 3, 'timesteps': 100000, 'seed': 2021}
    '''environment parameters'''
    params_env = {'num_a': 100, 'dim_s': 100, 'dim_a': 100, 'dim_feature': 10,
                  's_low': -1, 's_high': 1, 'a_low': -1, 'a_high': 1, 'r_arch': [10,10,10]}
    '''rl parameters'''
    params_rl = {'algos': ['A2C', 'DQN', 'PPO'], 'net_arch': [32,32,32], 'num_sim': 3}
    """
    '''setup and run experiments'''
    config = yaml.load(open('./config.yaml'))
    exp = ExperimentDataGenerator(config['params_exp'], config['params_env'], config['params_rl'])
    exp.run_experiments()

