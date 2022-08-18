import numpy as np
from cost_functions import *
from Activations import *

def backward_propagation(parameters, cache, activations, cost_function,tr_labels):
    model_activations, final_activation = activations
    grads = {}
    d_A_i = costs_grads[cost_function](cache['pred'],tr_labels)
    layers = len(parameters.keys())//2
    dZ = compute_gradient(d_A_i, cache['A' + str(layers - 1)], cache['Z' + str(layers)], grads, final_activation, layers)
    for i in range(layers - 1,0,-1):
        #backpropagate for layer i, from dZ(i+1) to dZ
        d_A_i = parameters['W'+ str(i+1)].T @ dZ
        #compute gradients
        dZ = compute_gradient(d_A_i, cache['A' + str(i-1)], cache['Z' + str(i)], grads, model_activations, i)
    return grads


def compute_gradient(dA, A, Z, grads,  function_activation,layer):
    'dA -> dZ'
    dZ = dA * activations_grads[function_activation](Z)
    grads['W' + str(layer)] = dZ @ A.T
    grads['b' + str(layer)] = np.sum(dZ,axis = 1, keepdims = True)
    return dZ
    


