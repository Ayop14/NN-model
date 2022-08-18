import numpy as np

def random_weight_initialization(units,input_shape):
    parameters = {'W1': np.random.rand(units[0], input_shape) * 0.01}
    parameters['b1'] = np.random.rand(units[0],1) * 0.01
    for i in range(1,len(units)):
        parameters['W'+  str(i+1)] = np.random.rand(units[i],units[i-1]) * 0.01
        parameters['b' + str(i+1)] = np.random.rand(units[i],1) * 0.01
    return parameters