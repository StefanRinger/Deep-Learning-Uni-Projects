import numpy as np

# fan_in = dimension of previous layer
# weights_shape = 'dimension' of current layer
# fan_out = dimension of next layer

class Constant: 
    def __init__(self, constant_value = 0.1):

        ''' returns a tensor that is filled with a constant value (given by weigth_initializer)'''
        
        self.Constant = constant_value

    def initialize(self, weights_shape, fan_in, fan_out):
        self.InitilaizedTensor = np.ones(weights_shape) * self.Constant 

        # what do we do with the fan_in and fan_out???
        return self.InitilaizedTensor

class UniformRandom:

    ''' returns a tensor that is filled with uniform [0,1] values '''

    def __init__(self):
        self.InitilaizedTensor = None

    def initialize(self, weights_shape, fan_in, fan_out):
        self.InitilaizedTensor =  np.random.random(weights_shape)
        return self.InitilaizedTensor  

class Xavier:
    def __init__(self):
        self.InitilaizedTensor = None

    def initialize(self, weights_shape, fan_in, fan_out):
        variance = 2 / (fan_in + fan_out)
        sigma  = np.sqrt(variance)
        mu = 0 #zero mean Guassian
        self.InitilaizedTensor  = np.random.normal(mu, sigma, weights_shape)
        return  self.InitilaizedTensor

class He:
    def __init__(self):
        self.InitilaizedTensor = None

    def initialize(self, weights_shape, fan_in, fan_out):
        variance = 2 /fan_in
        sigma  = np.sqrt(variance)
        mu = 0 #zero mean Guassian
        self.InitilaizedTensor  = np.random.normal(mu, sigma, weights_shape)
        return  self.InitilaizedTensor


