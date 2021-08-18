import numpy as np
from Layers import Base,Helpers
import copy

class BatchNormalization(Base.BaseLayer):
    def __init__(self,channels) -> None:
        super(BatchNormalization, self).__init__(isTrainable=True)
        self.trainable = True
        self.channels = channels
        self.current_mean = np.zeros((1, self.channels))
        self.current_std = np.zeros((1, self.channels))
        self.batch_mean = np.zeros((1, self.channels))
        self.batch_std = np.zeros((1, self.channels))
        self.testing_phase = False
        self._gradient_weights = 0
        self._gradient_bias = 0
        self._optimizer = None
        self._optimizerbias = None
        self.initialize(None,None)

    def initialize(self, _, __):
        self.weights = np.ones((1, self.channels))
        self.bias = np.zeros((1, self.channels)) 

    def forward(self,input_tensor):
        self.temp_input_tensor = input_tensor

        if input_tensor.ndim == 4:
           self.input_tensor = self.reformat(input_tensor)
        else: 
           self.input_tensor = input_tensor

        self.batch_mean = np.mean(self.input_tensor , axis=0)
        self.batch_std = np.std(self.input_tensor , axis=0)

        if not self.testing_phase:
            alpha = 0.8
            self.current_mean =  alpha * self.current_mean +  self.batch_mean * (1 - alpha)
            self.current_std = alpha * self.current_std +  self.batch_std * (1 - alpha)
            self.computed_input_tensor = (self.input_tensor - self.batch_mean)/ np.sqrt(self.batch_std **2 + np.finfo(float).eps)
        else:
            self.computed_input_tensor = (self.input_tensor - self.current_mean)/ np.sqrt(self.current_std**2 + (np.finfo(float).eps))
            

        output_layer = self.weights * self.computed_input_tensor + self.bias
        
        if self.temp_input_tensor.ndim == 4:
           output_layer = self.reformat(output_layer)

        return output_layer 

    def reformat(self, format_tensor):
        if format_tensor.ndim == 4:
           batch =  format_tensor.shape[0]
           channel = format_tensor.shape[1]
           rows = format_tensor.shape[2]
           columns = format_tensor.shape[3]

           format_tensor = np.reshape(format_tensor, (batch,channel, rows*columns)) 
           format_tensor = np.transpose(format_tensor,(0, 2, 1))
           format_tensor = np.reshape(format_tensor, (batch*rows*columns,  channel)) 

        else:  
            batch =  self.temp_input_tensor.shape[0]
            channel = self.temp_input_tensor.shape[1]
            rows = self.temp_input_tensor.shape[2]
            columns = self.temp_input_tensor.shape[3]
            format_tensor = np.reshape(format_tensor, (batch, rows*columns ,channel)) 
            format_tensor = np.transpose(format_tensor,(0, 2, 1))
            format_tensor = np.reshape(format_tensor, (batch ,  channel, rows, columns)) 

        return format_tensor




    def backward(self,error_tensor):
        temp_error_tensor = error_tensor
        
        if(error_tensor.ndim == 4):
            error_tensor = self.reformat(error_tensor)
        #else:
          #  error_tensor = np.reshape(error_tensor,self.computed_input_tensor.shape)    
       
        #Gradient with respect to weights
        weight_gradient =   np.sum(error_tensor * self.computed_input_tensor, axis = 0) 
        self.gradient_weights = np.reshape(weight_gradient, (1,self.channels))

        #Gradient with respect to weights
        weight_bias =   np.sum(error_tensor , axis=0) 
        self.gradient_bias = np.reshape(weight_bias, (1,self.channels))
        
        # gradient with respect to the input
        gradient_input = Helpers.compute_bn_gradients(error_tensor, self.input_tensor, self.weights, self.batch_mean, self.batch_std**2, np.finfo(float).eps)

         #update
        if self._optimizer is not None:
            self.weights = self._optimizer.calculate_update(self.weights, self.gradient_weights)
        if self._optimizerbias is not None:
            self.bias = self._optimizerbias.calculate_update(self.bias, self.gradient_bias)    

        if(temp_error_tensor.ndim == 4):
            gradient_input = self.reformat(gradient_input)

        return  gradient_input


     # gradient_weights property
    @property
    def gradient_weights(self):
        return self._gradient_weights
    @gradient_weights.setter
    def gradient_weights(self, value):
        self._gradient_weights = value
    # gradient_bias property
    @property
    def gradient_bias(self):
        return self._gradient_bias
    @gradient_bias.setter
    def gradient_bias(self, value):
        self._gradient_bias = value

    @property
    def optimizer(self):
        return self._optimizer
    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = copy.deepcopy(value)
        self._optimizerbias = copy.deepcopy(value)