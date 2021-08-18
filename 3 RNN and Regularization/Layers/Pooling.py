import numpy as np
from Layers import Base


class Pooling(Base.BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        super(Pooling, self).__init__()
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        self.max_value_list = None
        self.temp_tensor = None


    def initialize(self,  weights_initializer, bias_initializer):
        ''' init makes no sense since this layer is non parametric but neural network script calls this function for every appended layer'''
        pass
        return

    
    def forward(self, input_tensor):
        batch_size, channel_size, x_size, y_size = input_tensor.shape
        if(self.stride_shape[1] % 2 == 0):
            y_dim = int (np.floor(y_size / self.stride_shape[1]))
        else:
            y_dim = int (np.floor(y_size / self.stride_shape[1]) - (self.pooling_shape[1] / 2) )
        x_dim = int ( np.floor(x_size / self.stride_shape[0]))
        output_shape = (batch_size , channel_size , x_dim , y_dim)
        output_tensor = np.ones(output_shape)
        x = 0 
        y = 0  
        

        self.temp_tensor = np.zeros(input_tensor.shape)
        self.temp_tensor = input_tensor    

        self.max_value_list = np.zeros(output_shape)

    

        if(not (self.stride_shape == (1,1) and self.pooling_shape == (1,1))):
            for batch in range(batch_size):
                for channels in range(channel_size):
                    i = 0
                    for x in range(0, x_size, self.stride_shape[0]):
                        j = 0
                        if(i < x_dim): 
                            for y in range(0, y_size, self.stride_shape[1]):
                                if(j < y_dim):
                                    strided_list = input_tensor[batch, channels,  x: x + self.pooling_shape[0], y: y + self.pooling_shape[1]] 
                                    maximum_value =  np.max(strided_list)
                                    self.max_value_list[batch, channels, i , j] = maximum_value
                                j = j + 1
                            i = i + 1
            output_tensor = self.max_value_list.copy()
        else:
            output_tensor = input_tensor.copy()  
        return output_tensor 
        

    def backward(self, error_tensor):
        batch_size, channel_size, x_size, y_size = self.temp_tensor.shape
        previous_layer_error_tensor = np.zeros(self.temp_tensor.shape)

        for batch in range(batch_size):
            for channels in range(channel_size):
                i = 0
                for x in range(0, x_size-1, self.stride_shape[0]):
                    j = 0 
                    for y in range(0, y_size-1, self.stride_shape[1]):
                        strided_list = self.temp_tensor[batch, channels, x: x + self.pooling_shape[0], y: y + self.pooling_shape[1]] 
                        max_value = np.max(strided_list)
                        strided_list = np.where(strided_list == max_value, 1, strided_list)
                        strided_list = np.where(strided_list != 1,  0, strided_list)
                        previous_layer_error_tensor[batch, channels, x: x + self.pooling_shape[0], y: y + self.pooling_shape[1]] += strided_list * error_tensor[batch, channels, i,j]
                        j = j + 1
                    i = i + 1   
                         
        return  previous_layer_error_tensor