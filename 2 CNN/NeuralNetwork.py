import copy
from Layers import SoftMax

class NeuralNetwork:

    def __init__(self, optimizer, weights_initializer, bias_initializer ):

        self.optimizer = optimizer
        self.loss = []  # contains the loss value for each iteration after calling train
        self.layers = []    # holds the architecture
        self.data_layer = None  # provides input data and labels
        self.loss_layer = None  # special layer providing loss and prediction
        self.weights_initializer = weights_initializer   # these are functions!
        self.bias_initializer = bias_initializer # these are functions!



    
    def forward(self):

        # get input & label data by calling data_layer.next()
        input_tensor, label_tensor = self.data_layer.next()
        self.label_tensor = label_tensor
        
        forward_tensor = input_tensor

        # loop over the list of layers. Call next() on each layer and pass into successive layer until done. Last layer = loss layer
        for i in range(len(self.layers)):
            forward_tensor = self.layers[i].forward(forward_tensor)

        # call loss layer forward, that is stored separately:

        loss = self.loss_layer.forward(forward_tensor, label_tensor)

        return loss



    def backward(self):

        ''' Propagate the error backwards through the network
        Input : None
        Output : None
        '''

        backward_tensor = self.loss_layer.backward(self.label_tensor) # backpropagate through loss layer

        # backpropagate through other layers:

        for i in reversed(range(len(self.layers))): # iterate through layers backwards
            backward_tensor = self.layers[i].backward(backward_tensor)

        return



    def append_layer(self, layer):

        deep_copied_optimizer = copy.deepcopy(self.optimizer)   # for storing values like momentum for each individual layer
        layer.optimizer = deep_copied_optimizer # thank you, @property

        # initialize the layers with the attached initializer functions that are given to this network description script
        layer.initialize(self.weights_initializer, self.bias_initializer)
        
        self.layers.append(layer)




    def train(self, iterations: int):

        for i in range(iterations):

            self.loss.append(self.forward()) # call forward -> returns loss
            self.backward()

        return


    def test(self, input_tensor):

        ''' propagates the input tensor through the network and returns the prediction of the last layer.
        For classification tasks we typically query the probabilistic output of the SoftMax layer.
        Input: input_tensor
        '''

        
        forward_tensor = input_tensor

        # loop over the list of layers. Call next() on each layer and pass into successive layer until done
        for i in range(len(self.layers)):
            forward_tensor = self.layers[i].forward(forward_tensor)

        # call last layer: SoftMax
        probability_layer = SoftMax.SoftMax()
        predictions = probability_layer.forward(forward_tensor)

        return predictions




