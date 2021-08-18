
from Layers import Base
import numpy as np

class Dropout(Base.BaseLayer):


    def __init__(self, probability):

        '''Implements an inverted dropout layer according to the given probability parameter.'''
        super(Dropout, self).__init__()
        self.probability = probability # fraction units to keep = p. We rescale all the activations by inverse probability q=1/p to avoid getting large activations
        self.trainable = False # layer has no adjustable parameters
        self.testing_phase = None
        self.dropout_matrix = None

        


    # distinguish training and testing phase
    # self.testing_phase


    def initialize(self):

        pass # not trainable


        return





    def forward(self, input_tensor):
        # for training phase: Rescale

        #scaling_factor = float (1.0 / self.probability)
        #print(input_tensor)

        if not self.testing_phase:   # training phase        

            # if we remove elements then we need to rescale whole matrix
            self.dropout_matrix = np.random.binomial(1, self.probability, size=input_tensor.shape).astype(np.float) / self.probability # success probability p -> 1-p set every with 0
            output_tensor =  input_tensor * self.dropout_matrix     # rescaling * on/off matrix * input

        else:

            output_tensor =  input_tensor # due to inverted dropout we get rid of multiplication with p in test time



        return output_tensor



    def backward(self, error_tensor):
        # for testing phase: Do nothing (in classical dropout here the keep probabilty...)

        output_tensor = error_tensor * self.dropout_matrix


        return output_tensor