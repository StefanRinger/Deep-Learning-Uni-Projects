import numpy as np
from Layers import Base

class ReLU(Base.BaseLayer):
    def __init__(self):
        super(ReLU, self).__init__()
        self.input = None
        return
    
    def forward(self, input_tensor):
        input_tensor[input_tensor <= 0] = 0 # term in bracket creates binary mask of spots where the tensor/error smaller than 0. The error of these spots are then set to 0 
        self.input = input_tensor
        return input_tensor

    def backward(self, error_tensor):     
        error_tensor[self.input <= 0] = 0
        return error_tensor