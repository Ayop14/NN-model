import numpy as np
from Weight_initialization import *
from Forward_propagation import *
from Backward_propagation import *
from cost_functions import * 

def update_parameters(parameters,grads, lr):
    for i in range(len(parameters.keys())//2):
        W = 'W' + str(i)
        b = 'b' + str(i)
        parameters[W] = parameters[W] - lr * grads[W]
        parameters[b] = parameters[b] - lr * grads[b]

#Vectorized implementation of the binary NN model
def Binary_NN_Model(units, iter, activations, tr_data, tr_lab, ts_data,ts_lab, lr = 0.001):
    '''Input:
        units: list indicating number of units for each layer 
        iter: n of iterations optimizing the model
        activations: tuple of str for model and output activations ('relu', 'sigmoid')
        numpy data matrices for the model
        lr: Learning rate of the model
       Output:
       Training error, test error
    '''
    assert units[-1] == 1, 'Last layer must have 1 unit por BNN'
    parameters = random_weight_initialization(units, tr_data.shape[0])
    cost_record = []
    #training
    for _ in range(iter):
        print('iter------------------')
        breakpoint()
        pred, cache = forward_propagation(parameters,tr_data, activations)
        cost = log_reg_cost(pred, tr_lab)
        cost_record.append(cost)
        parameters_grads = backward_propagation(parameters, cache, activations, 'log_reg',tr_lab)
        update_parameters(parameters, parameters_grads, lr)
    #testing
    test_pred, _ = forward_propagation(parameters,ts_data, activations)
    tr_pred = forward_propagation(parameters,tr_data, activations)
    test_pred = test_pred >= 0.5
    tr_pred = tr_pred >= 0.5
    test_err = np.sum(test_pred != ts_lab)/ len(ts_lab)
    trai_err = np.sum(tr_pred != tr_lab)/ len(ts_lab)
    return test_err, trai_err


