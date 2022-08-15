import numpy as np
from Activations import * 
def forward_propagation(parameters,tr_data, activations):
    X = tr_data
    cache = {}
    model_activation, final_activation = activations
    for i in range(len(parameters.keys())/2 - 1):
        X = np.transpose(parameters['W' + str(i+1)]) @ X + parameters['b' + str(i + 1)]
        X = activation(model_activation)(X)
        cache['a' + str(i+1)] = X
    X = np.transpose(parameters['W' + str(i+1)]) @ X + parameters['b' + str(i + 1)]
    X = activation(final_activation)(X)
    cache['a' + str(i+1)] = X
    return X, cache