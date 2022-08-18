import numpy as np

def activation(activation):
    aux = activation.lower()
    return activations[aux]

def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def grad_sigmoid(X):
    aux = sigmoid(X)
    return aux * (1 - aux)

def relu(X):
    return np.maximum(X, 0)

def grad_relu(X):
    return np.sum(X >= 0, axis = 0, keepdims = True)

activations = {'sigmoid' : sigmoid, 'relu' : relu}
activations_grads = {'sigmoid' : grad_sigmoid, 'relu' : grad_relu}