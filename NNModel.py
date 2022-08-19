import numpy as np
from Weight_initialization import *
from Forward_propagation import *
from Backward_propagation import *
from cost_functions import * 
import copy

def update_parameters(parameters,grads, learning_rate = 0.001):
    res = {}
    for i in range(1,len(parameters.keys())//2 + 1):
        Wi = 'W' + str(i)
        bi = 'b' + str(i)
        W = copy.deepcopy(parameters['W' + str(i)])
        b = parameters['b' + str(i)]
        res[Wi] = W - learning_rate * grads[Wi]
        res[bi] = b - learning_rate * grads[bi]
    return res

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
    wat = []
    wat2 = []
    #training
    for _ in range(iter):
        print('iter------------------')
        pred, cache = forward_propagation(parameters,tr_data, activations)
        cost = log_reg_cost(pred, tr_lab)
        cost_record.append(cost)
        parameters_grads = backward_propagation(parameters, pred, cache, activations, 'log_reg',tr_lab)
        wat.append(parameters['W1'][0,0])
        wat2.append(parameters_grads['W1'][0,0])
        parameters = update_parameters(parameters, parameters_grads, learning_rate = 0.008)
    #testing
    breakpoint()
    ts_pred, _ = forward_propagation(parameters,ts_data, activations)
    tr_pred,_ = forward_propagation(parameters,tr_data, activations)
    ts_pred = ts_pred >= 0.5
    tr_pred = tr_pred >= 0.5
    test_err = np.sum(ts_pred == ts_lab)/ len(ts_lab[0])
    trai_err = np.sum(tr_pred == tr_lab)/ len(tr_lab[0])
    return test_err, trai_err


