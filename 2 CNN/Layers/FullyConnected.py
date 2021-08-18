import numpy as np
from numpy.core.arrayprint import _none_or_positive_arg
from Layers import Base
from Optimization import Optimizers

class FullyConnected(Base.BaseLayer):
    def __init__(self,input_size, output_size):
        super(FullyConnected, self).__init__(isTrainable=True)
        self.input_size = input_size
        self.output_size = output_size
        self.weights = None
        
        self.weights = np.random.rand(self.input_size + 1,self.output_size) #weights of the current layer uniformly random in the range [0, 1)
        ## Actually this should be done by the intialization function but somehow it does not work all the time????
        
        self._optimizer = None
        self._gradient_weights = None
        self.bias = None

    def initialize(self,  weights_initializer, bias_initializer):
        shape = (self.output_size, self.input_size)
        self.weights = weights_initializer.initialize(self.weights.shape, shape[1], shape[0])
        #self.weights = weights_initializer.initialize(weights_shape=shape, fan_in=shape[1], fan_out=shape[0])
        shape = (self.output_size, 1) # bias is a vector of length = ouput nodes
        self.bias = bias_initializer.initialize(shape, shape[1], shape[0]) # correct ?????
        return



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
        #self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
        if (self._optimizer):
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
        return  output_error_tensor