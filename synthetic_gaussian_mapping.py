
import numpy as np
np.set_printoptions(precision=4)


class SyntheticGaussianMapping:
    '''generate synthetic feature extractor'''

    def __init__(self, params):
        self.__dict__.update(params)
        self.initialize_weights()
        self.activation = lambda z: np.exp(-z**2)

    def initialize_weights(self):
        '''initialize the network from normal distribution'''
        np.random.seed(self.seed)
        self.dims = [self.dim_in, *self.dim_layers, self.dim_out]
        self.num_layers = len(self.dims) - 1
        self.weights = {}
        for l in range(self.num_layers):
            self.weights[l] = self.weight_norm * np.random.randn(self.dims[l] + 1, self.dims[l+1])

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

    def report(self):
        '''display the mapping info'''
        print(f'\ncreated synthetic map F: R^{self.dim_in} --> R^{self.dim_out}')
        print('network architecture:    ', end='')
        print(*self.dims[1:], sep=' --- ')
        print(f'network weight shapes:   ', end='')
        print(*[f'{w.shape[0]}x{w.shape[1]}' for w in self.weights.values()], sep=' -- ', end='\n\n')
        map_input = np.random.randn(3, self.dim_in)
        map_output = self.propagate(map_input)
        for i in range(3):
            print(f'random input {i}: {map_input[i]}')
            print(f'output {i}:       {map_output[i]}')


if __name__ == '__main__':
    '''configure and create the mapping'''
    params_map = {'dim_in': 6, 'dim_out': 4, 'dim_layers': [10,20], 'weight_norm': 1.0, 'seed': 0}
    F = SyntheticGaussianMapping(params_map)
    F.report()
