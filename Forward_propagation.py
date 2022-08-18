import numpy as np
from Activations import * 
def forward_propagation(parameters,tr_data, activations):
    X = tr_data
    cache = {}
    model_activation, final_activation = activations
    for i in range(len(parameters.keys())//2 - 1):
        X = parameters['W' + str(i+1)] @ X + parameters['b' + str(i + 1)]
        cache['Z' + str(i+1)] = X
        X = activation(model_activation)(X)
        cache['A' + str(i+1)] = X
    #We add 2 bc 1 of index stride (starts in 0) and 1 of what would be last iteration
    X = parameters['W' + str(i+2)] @ X + parameters['b' + str(i + 2)]
    cache['Z' + str(i+2)] = X
    X = activation(final_activation)(X)
    cache['pred'] = X
    return X, cache