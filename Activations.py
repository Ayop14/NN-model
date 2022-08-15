import numpy as np

def activation(activation):
    aux = activation.lower()
    return activations[aux]

def softmax(x):
    #acuerdate de añadir al diccionario activationsla variable y el valor
    pass

def relu(x):
    pass

activations = {'softmax' : softmax, 'relu' : relu}