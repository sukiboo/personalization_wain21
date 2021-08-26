
import numpy as np


class SyntheticGaussianMapping:
    '''generate synthetic feature extractor'''

    def __init__(self, params_map):
        self.__dict__.update(params_map)
        self.initialize_weights()
        self.activation = lambda z: np.exp(-z**2)

    def initialize_weights(self):
        '''initialize the network from normal distribution'''
        np.random.seed(self.seed)
        self.dims = [self.dim_in, *self.dim_layers, self.dim_out]
        self.num_layers = len(self.dims) - 1
        self.weights = {}
        for l in range(self.num_layers):
            self.weights[l] = np.random.randn(self.dims[l] + 1, self.dims[l+1])

    def propagate(self, x):
        '''propagate input through the network'''
        z = np.array(x, ndmin=2)
        for l in range(self.num_layers):
            z = np.concatenate([np.ones((z.shape[0],1)), z], axis=1)
            if l < self.num_layers - 1:
                z = self.activation(np.matmul(z, self.weights[l]))
            else:
                z = np.tanh(np.matmul(z, self.weights[l]))
        return z

