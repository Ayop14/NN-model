import numpy as np

def random_weight_initialization(units,input_shape):
    parameters = {'W1': np.random.random((units[0], input_shape))}
    parameters['b1'] = np.random.random((units[0],1))
    for i in range(1,len(units)):
        parameters['W'+  str(i+1)] = np.random.random((units[i],units[i-1]))
        parameters['b' + str(i+1)] = np.random.random((units[i],1))
    breakpoint()
    return parameters