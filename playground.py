import numpy as np
from NNModel import *


data = np.genfromtxt('fraud_detection_bank_dataset.csv', delimiter=',', skip_header = 1)

#preparing dataset
np.random.seed(3)
N = data.shape[0]
perm = np.random.permutation(N)
data = data[perm].T

tr_data = data[:N,:int(0.9 * N)]
tr_lab = data[-1, np.newaxis,:int(0.9 * N)]
ts_data = data[:N,int(0.9 * N):]
ts_lab = data[-1, np.newaxis,int(0.9 * N):]

test_err, trai_err = Binary_NN_Model([1], 100, ('sigmoid','sigmoid'), tr_data, tr_lab, ts_data,ts_lab, lr = 0.008)
print(test_err, trai_err)