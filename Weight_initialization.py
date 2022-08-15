import numpy as np

def random_weight_initialization(units,input_shape):
    parameters = {'W1': np.random(input_shape,units[0])}
    parameters['b1'] = np.random(input_shape,1)
    for i in range(1,len(units)-1):
        parameters['W'+  str(i+1)] = np.random(units[i],units[i-1])
        parameters['b' + str(i+1)] = np.random(units[i],units[i-1])
    return parameters