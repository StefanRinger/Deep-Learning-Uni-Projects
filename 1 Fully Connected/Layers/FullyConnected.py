import numpy as np
from Layers import Base
from Optimization import Optimizers

class FullyConnected(Base.BaseLayer):
    def __init__(self,input_size, output_size):
        super(FullyConnected, self).__init__(True)
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.rand(input_size + 1,output_size) #weights of the current layer uniformly random in the range [0, 1)
        self.input_tensor = [] # coming as a parameter in forward method 
        self._optimizer = None
        self._gradient_weights = None

    def forward(self, input_tensor):
        #according to the slides 13 adding one more column in input.   
        input_tensor = input_tensor.T
        column_number = input_tensor.shape[1]
        new_column = np.ones(column_number) 
        self.input_tensor = np.vstack((input_tensor, new_column))
        output_layer = np.dot((self.input_tensor.T), self.weights)
        return output_layer

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value
    
    @property
    def gradient_weights(self):  
        return  self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self,value):
        self._gradient_weights = value

    def backward(self, error_tensor):
        output_error_tensor = np.dot(error_tensor,self.weights.T)  #According to the formula on slide 24
        output_error_tensor = np.delete(output_error_tensor, (-1), axis=1)  #deleting last row of the error tensor -> bias
        self._gradient_weights = np.dot(self.input_tensor, error_tensor) #calculating gradient tensor to update the weight
        if (self._optimizer):
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
        return  output_error_tensor