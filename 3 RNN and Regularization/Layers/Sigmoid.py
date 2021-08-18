import numpy as np
from Layers.Base import BaseLayer

class Sigmoid(BaseLayer):
    def __init__(self):
        super(Sigmoid, self).__init__()
        self.activations = None

    def forward(self, input_tensor):
        self.activations = np.divide(1, np.add(1, np.exp((-1 * input_tensor))))
        return self.activations.copy()    

    def backward(self, error_tensor):
        return np.multiply(error_tensor, np.multiply(self.activations, np.subtract(1, self.activations)))    