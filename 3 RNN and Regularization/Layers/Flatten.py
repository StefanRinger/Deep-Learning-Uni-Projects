import numpy as np
from Layers import Base


class Flatten(Base.BaseLayer):
    def __init__(self):
        super(Flatten, self).__init__()
        self.inputShape = None
        pass

    def initialize(self,  weights_initializer, bias_initializer):
        ''' init makes no sense since this layer is non parametric but neural network script calls this function for every appended layer'''
        pass
        return


    def forward(self,input_tensor):
        self.inputShape = input_tensor.shape
        # input shape is eg because of CNN: (batchsize, input_channels or feature_maps, input_x_data, input_y_data)
        # if we want to connect this to a Fully connected layer we need to flatten/make 1d-Arrays out of the multi-dim spacial tensors
        # output shape should be just (batchsize , output_nodes)
        # with output_nodes = input_channels * input_x_data * input_y_data etc
        if len(input_tensor.shape)==2:
            return input_tensor
        flattenedOutput = input_tensor.reshape(self.inputShape[0], self.inputShape[1] * self.inputShape[2] * self.inputShape[3]) # could be made more flexible. Eg if we did video we would have rank 5 and not rank 4 tensors.
        return flattenedOutput

    def backward(self, error_tensor):
        originalShape = error_tensor.reshape(self.inputShape)    
        return originalShape


