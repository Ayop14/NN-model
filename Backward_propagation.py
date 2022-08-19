import numpy as np
from cost_functions import *
from Activations import *

def backward_propagation(parameters, pred, cache, activations, cost_function,tr_labels):
    model_activations, final_activation = activations
    grads = {}
    d_A_i = costs_grads[cost_function](pred,tr_labels)
    layers = len(parameters.keys())//2
    dZ = compute_gradient(d_A_i, cache['A' + str(layers - 1)], cache['A' + str(layers)], grads, final_activation, layers)
    for i in range(layers - 1,0,-1):
        #backpropagate for layer i, from dZ(i+1) to dZ
        d_A_i = parameters['W'+ str(i+1)].T @ dZ
        #compute gradients
        dZ = compute_gradient(d_A_i, cache['A' + str(i-1)], cache['A' + str(i)], grads, model_activations, i)
    return grads


def compute_gradient(dA, Ai_1, Ai, grads,  function_activation,layer):
    'dA -> dZ'
    dZ = dA * activations_grads[function_activation](Ai)
    m = Ai.shape[1]
    grads['W' + str(layer)] = dZ @ Ai_1.T / m
    grads['b' + str(layer)] = np.sum(dZ,axis = 1, keepdims = True) / m
    return dZ
    


