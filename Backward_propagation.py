import numpy as np
from cost_functions import *
from Activations import *

def backward_propagation(parameters, cache, activations, cost_function,tr_labels):
    model_activations, final_activation = activations
    grads = {}
    d_A_i = costs_grads(cost_function)(parameters['pred'],tr_labels)

    for i in range(len(parameters.keys)/2,0,-1):
        dZ = compute_gradient(d_A_i, cache['A' + str(i - 1)], cache['Z' + str(i)], grads, model_activations, i)
        #backpropagate
        d_A_i = parameters['W'+ str(i)] @ dZ
        # This still makes a backpropagation that we do not need----------------
    return grads


def compute_gradient(dA, A, Z, grads,  function_activation,layer):
    'dA -> dZ'
    dZ = dA * activations_grads(function_activation)(Z)
    grads['dW' + str(layer)] = dZ @ A.T
    grads['db' + str(layer)] = np.sum(dZ,axis = 1, keepdims = True)
    return dZ
    


