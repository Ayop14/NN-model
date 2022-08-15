import numpy as np

def activation(activation):
    aux = activation.lower()
    return activations[aux]

def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def relu(X):
    return np.maximum(X, 0)

activations = {'softmax' : sigmoid, 'relu' : relu}