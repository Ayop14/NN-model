import numpy as np
from cost_functions import *

def backward_propagation(parameters, cache, activations, cost_grad_function):
    grad = costs(cost_grad_function)