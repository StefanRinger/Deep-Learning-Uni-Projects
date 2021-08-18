import numpy as np
from Layers import Base

np.seterr(all='ignore')
class SoftMax(Base.BaseLayer):
    def __init__(self):
        super(SoftMax, self).__init__()
        self.prediction_tensor = None
        return
    
    def forward(self, input_tensor):
        #Tutorial 1, slides 12-13
        #Adjusting the values to avoid infinity: if x large -> e^x very large!
        maxValue = np.max(input_tensor)
        shifted_input_tensor = np.subtract(input_tensor, maxValue) 
        #Softmax Activation Function
        exp_tensor = np.exp(shifted_input_tensor)
        self.prediction_tensor = exp_tensor/exp_tensor.sum(axis=1,keepdims=True) # normalize by sum over batch
        return self.prediction_tensor #output = prediction/activation that is forwarded

    def backward(self, error_tensor):
        #Tutorial 1, slide 14
        output_error_tensor = self.prediction_tensor * (error_tensor - (error_tensor * self.prediction_tensor).sum(axis=1,keepdims=True))
        return output_error_tensor