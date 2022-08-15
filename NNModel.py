import numpy as np
from Weight_initialization import *
from Forward_propagation import *
from Backward_propagation import *
from cost_functions import * 

#Vectorized implementation of the binary NN model
def Binary_NN_Model(units, iter, tr_data, tr_lab, ts_data,ts_lab, lr = 0.001):
    '''Input:
        units: list indicating number of units for each layer 
        iter: n of iterations optimizing the model
        numpy data matrices for the model
        lr: Learning rate of the model
       Output:
       Training error, test error
    '''
    assert units[-1] == 1, 'Last layer must have 1 unit por BNN'
    parameters = random_weight_initialization(units, tr_data.shape[1])
    cost_record = []
    for _ in range(iter):
        pred = forward_propagation(parameters,tr_data)
        cost = compute_cost(pred, tr_lab)
        cost_record.append(cost)
        parameters = backward_propagation()
