import numpy as np

def log_reg_cost(pred, tr_lab):
    cost = np.sum(tr_lab * np.log(pred + 1e-6) + (1 - tr_lab) * np.log(1-pred + 1e-6))
    return  -1/tr_lab.shape[1] * cost

def log_reg_cost_grad(A, Y):
    return -Y/A + (1-Y)/(1-A)

costs = {'log_reg': log_reg_cost}
costs_grads = {'log_reg': log_reg_cost_grad}