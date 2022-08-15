import numpy as np

def log_reg_cost(pred, tr_lab):
    cost = np.sum(tr_lab * np.log(pred) + (1 - tr_lab) * np.log(1-pred))
    return  -1/tr_lab.shape[1] * cost