
from Layers import Base, TanH, Sigmoid
from Layers import *

import copy

import numpy as np




class RNN(Base.BaseLayer):

    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.trainable = True

        self.input_size = input_size  # scalar
        self.hidden_size = hidden_size  # scalar
        self.hidden_state = np.zeros(hidden_size)  # vector
        self.output_size = output_size  # scalar

        self._memorize = False  # Does RNN regards subsequent sequences belonging to same long sequence?
        # in the description it says false but only works with true

        self.optimizer = None

        ###### in out sizes!!!!!!

        self.FC1 = FullyConnected.FullyConnected(hidden_size + input_size, hidden_size)
        self.TanH = TanH.TanH()
        self.FC2 = FullyConnected.FullyConnected(hidden_size, output_size)
        self.Sigmoid = Sigmoid.Sigmoid()

        self.gradient_weights = copy.copy(
            self.FC1._gradient_weights)  # make shallow copy (pointer) of gradient weights of FC1. Because property doesn't work.....
        self.weights = copy.copy(
            self.FC1.weights)  # make shallow copy (pointer) of gradient weights of FC1. Because property doesn't work.....

    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, value_boolean):
        self._memorize = value_boolean

    ##### some weird stuff with this and has no .FC1 going on

    @property
    def gradient_weights(self):
        return  self.FC1._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self,value):
        self.FC1._gradient_weights = value



    @property
    def weights(self):
        return self.FC1.weights

    @weights.setter
    def weights(self, value):
        self.FC1.weights = value

    ##### some weird stuff with this and the backwards step going on. not such a clean solution but works somehow

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    def initialize(self, weights_initializer, bias_initializer):

        # Weights for FC1 are considered to be the weights for the RNN
        self.FC1.initialize(weights_initializer, bias_initializer)
        self.weights = self.FC1.weights
        self.bias = self.FC1.bias

        self.FC1.initialize(weights_initializer, bias_initializer)
        return

    def forward(self, input_tensor):

        # batch dimension = time dimension

        if self._memorize == False:
            hidden_state_previous = np.zeros(
                self.hidden_size)  # set hidden state to zero for comming time indices next iterations. Correct? or mean between forward calls???
        else:
            hidden_state_previous = self.hidden_state

        batch_size = int(input_tensor.shape[0])

        self.tanh_activations = np.zeros((batch_size, self.hidden_size)).astype(np.float)
        self.FC2_input = np.zeros((batch_size, self.hidden_size + 1)).astype(np.float)  # FC2
        self.sigmoid_activations = np.zeros((batch_size, self.output_size)).astype(np.float)
        self.FC1_input = np.zeros((batch_size, self.input_size + self.hidden_size + 1)).astype(
            np.float)  # FC1

        output_tensor = np.zeros((batch_size, self.output_size))  # output shape: time indices, output size

        for i in range(batch_size):  # loop over batch size / time steps

                ###### evtl ist das hier so gemeint, dass sich beide die selben weights teilen, dh wirklich stacken nicht concatenaten -> problem mit fehlender dimension weg

            stacked_input = np.concatenate((hidden_state_previous.flatten(), input_tensor[
                i]))  # 1 for bias is done in fc layer. appended at last position

            self.feed_tensor = np.zeros((1, stacked_input.shape[0]))
            self.feed_tensor[0] = stacked_input

            tanh_input = self.FC1.forward(
                self.feed_tensor)  # the weights of this layer are considered to be the weights for the entire class
            self.FC1_input[i] = self.FC1.input_tensor.copy().flatten()

            current_hidden_state = self.TanH.forward(tanh_input)

            #self.hidden_state = self.TanH.forward(tanh_input)  # save hidden state for next iteration
            self.tanh_activations[i] = self.TanH.activations.copy()

            #simgoid_input = self.FC2.forward(self.hidden_state)
            simgoid_input = self.FC2.forward(current_hidden_state)

            hidden_state_previous = current_hidden_state

            self.FC2_input[i] = self.FC2.input_tensor.copy().flatten()

            output_vec = self.Sigmoid.forward(simgoid_input)
            self.sigmoid_activations[i] = self.Sigmoid.activations.copy()

            output_tensor[i] = output_vec  # stack vectors into the output tensor

            if self._memorize:
                #self.hidden_state = np.zeros(self.hidden_size)
                self.hidden_state = current_hidden_state

            """ why does forward state ful fail?? """

        return output_tensor

    def backward(self, error_tensor):

        batch_size = int(error_tensor.shape[0])  # ie number of time steps
        output_error = np.zeros((batch_size, self.input_size))

        hidden_states_gradient = np.zeros(
            self.hidden_size)  # first iteration: No previous hidden state gradient is available -> all zeros

        total_gradient_weights_FC1 = np.zeros(self.FC1.weights.shape).astype(np.float)
        total_gradient_weights_FC2 = np.zeros(self.FC2.weights.shape).astype(np.float)

        for i in range(batch_size - 1, -1, -1):  # go back in time ie batch_size= 100 -> 99,98,97, ... 2,1,0

            # sigmoid layer
            self.Sigmoid.activations = self.sigmoid_activations[i].reshape(1, -1)
            sigmoid_error = self.Sigmoid.backward(error_tensor[i])

            # Fully Connected Layer 2
            """ why does misaligned dimensions pop up here??? """

            self.FC2.input_tensor = self.FC2_input[i].reshape(1, -1).T  # restore the respective input tensor at the time of forward calling
            FC2_error = self.FC2.backward(sigmoid_error)
            # in backward pass of FC layer the gradient weights are computed. We call them and add them up to the total gradient weights of the batch for the layer FC2
            gradient_weights_FC2 = self.FC2.gradient_weights
            total_gradient_weights_FC2 += gradient_weights_FC2

            # inversion of copy is split as in tutorial explained. Gives linking to previous time steps through hidden state
            intermediate_error = FC2_error + hidden_states_gradient

            # TanH layer
            self.TanH.activations = self.tanh_activations[i].reshape(1, -1)  # restore activations
            TanH_error = self.TanH.backward(intermediate_error)

            # Fully Connected Layer 1
            self.FC1.input_tensor = self.FC1_input[i].reshape(1, -1).T # restore the respective input tensor at the time of forward calling
            FC1_error = self.FC1.backward(TanH_error).flatten()
            # in backward pass of FC layer the gradient weights are computed. We call them and add them up to the total gradient weights of the batch for the layer FC1
            gradient_weights_FC1 = self.FC1.gradient_weights
            total_gradient_weights_FC1 += gradient_weights_FC1

            # split up the concatenation now:
            # stacked_input = np.concatenate((hidden_state_previous.flatten(), input_tensor[i])) # 1 for bias is done in fc layer. appended at last position
            [hidden_states_gradient, input_error] = np.split(FC1_error,
                                                             [self.hidden_size])  # input error includes bias error

            output_error[i] = input_error

        self.gradient_weights = total_gradient_weights_FC1

        # now update the weights:
        # we have to only update the weights of the Fully Connected layers. We do not assign them an optimizer so that they don't do this after every backwards call
        # we need the inputs to forward for calculating the gradient

        # only update weights if optimizer is given
        if (self.optimizer):
            """ self.FC2._gradient_weights = np.dot(self.hidden_state, FC2_error.T) #calculating gradient tensor to update the weight
            #self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
            self.FC2.weights = self.optimizer.calculate_update(self.FC2.weights, self.FC2._gradient_weights)

            self.FC1._gradient_weights = np.dot(self.feed_tensor, FC1_error.T) #calculating gradient tensor to update the weight
            self.gradient_weights = self.FC1._gradient_weights
            #self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
            self.FC1.weights = self.optimizer.calculate_update(self.FC1.weights, self.FC1._gradient_weights) """

            self.FC1.weights = self.optimizer.calculate_update(self.FC1.weights.copy(), total_gradient_weights_FC1)
            self.FC2.weights = self.optimizer.calculate_update(self.FC2.weights.copy(), total_gradient_weights_FC2)
            self._weights = self.FC1.weights  # these are the weigths considered to be representative for the network as a whole

        return output_error



