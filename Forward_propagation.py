import numpy as np

def forward_propagation(parameters,tr_data, activations):
    X = tr_data
    for i in range(len(parameters.keys())/2):
        X = np.transpose(parameters['W' + str(i+1)]) @ X + parameters['b' + str(i + 1)]
    return X