import numpy as np
from Layers import Base

import copy
from scipy.ndimage import correlate
from scipy.ndimage import convolve
import scipy.signal



class Conv(Base.BaseLayer):


    def __init__(self, stride_shape, convolution_shape, num_kernels):

        '''
        Implements a convolutional layer.
        - stride shape: can be a single value or a tuple. The latter allows for different strides in the spatial dimensions.
        – convolution shape: determines whether this object provides a 1D or a 2D convolution layer. For 1D, it has the shape [c, m], whereas for 2D, it has the shape [c, m, n], where c represents the number of input channels, and m, n represent the spatial extent of the filter kernel.
        – num kernels: is an integer value.
        '''

        super(Conv, self).__init__(isTrainable=True)    # has trainable weights -> set inherited var to true
        
        # parameters of the layer

        # handle the stride cases to always give str_x and str_y:
        self.stride_shape = stride_shape # scalar or tuple
        self.is2D = (len(convolution_shape) == 3) #num elements in the tuple -> 1D or 2D ?

        self.stride_x = None
        if type(self.stride_shape) is int:  # 2d case where just 1 int is given, that is supposed to be taken for both spacial dimensions (square)
            self.stride_y = self.stride_x = self.stride_shape
        else:
            if self.is2D == True:   # general 2d stride case
                self.stride_y, self.stride_x = self.stride_shape[0], self.stride_shape[1]
            else:
                self.stride_y = self.stride_shape[0]   #1d case
        


        self.convolution_shape = convolution_shape #1D: [input_channels, y_coord], 2D: [channels, y_coord, x_coord]
        self.num_kernels = num_kernels # int
        self.num_channels = convolution_shape[0]

        #weights and bias
        self.weights = None
        self.bias = None

        #set parameters for weights and bias for external initalization function
        self.weights_shape = (num_kernels, *convolution_shape) # (feature_maps, input_channels,  x_coord, y_coord)
        #tutorial 2, slide 4:
        self.fan_in = np.prod(self.convolution_shape) # input_channels × kernel_height(y) × kernel_width(x)]
        self.fan_out = np.prod(self.convolution_shape[1:]) * self.num_kernels # [output_channels × kernel_height(y) × kernel_width(x)]

        # standardly intialize randomly [0,1[ uniform:
        self.weights = np.random.uniform(low=0.0, high=1.0, size=self.weights_shape) # number of weights = (num_kernels * input_channels) * (x_size * y_size) = feature maps * convolution size
        self.bias = np.random.uniform(low=0.0, high=1.0, size=self.num_kernels) # one bias for every feature map
        
        
        # gradient weighs & bias
        self._gradient_weights = None
        self._gradient_bias = None

        # seperate optimizers for weights and bias
        self._optimizer = None
        self.bias_optimizer = None



        #end of constructor of class






    



    
    def forward(self, input_tensor):

        '''
        The input layout for 1D is defined in b, c, y order, for 2D in b, c, y, x order. Here, b stands for the batch, c represents the channels and x, y represent the spatial dimensions.
        Recieves input tensor of layer and returns a tensor that serves as input for next layer.
        '''
        self.input_tensor = input_tensor # save for backward pass (shape!)



        # handle case stride being scalar or tuple
        # define stride variables



        output_tensor = None


        help_shape = None

        # loop over batch (input_tensor: first variable) -> channels & spacials remaining
        for batch_index, batch_image in enumerate(input_tensor):

            correlated_kernels = None

            # loop over kernels in weights
            for kernel_index, kernel in enumerate(self.weights):

                # handle 1 kernel 1Dx1D case: TO DO

                #IF: 2D Case
                # parameters: we pad with constant (mode) value (cval) 0s.
                # the strides implement the subsampling
                if self.is2D == True:
                    image_kernel_correlation = correlate(batch_image, kernel, mode='constant', cval=0)
                    #print(type(image_kernel_convolution))
                    image_kernel_correlation = image_kernel_correlation[int((self.num_channels)/2)][::self.stride_y, ::self.stride_x] + self.bias[kernel_index]

                #ELIF: 1D Case
                else:
                    image_kernel_correlation = correlate(batch_image, kernel, mode='constant', cval=0)
                    #print(type(image_kernel_convolution))
                    image_kernel_correlation = image_kernel_correlation[int((self.num_channels)/2)][::self.stride_y] + self.bias[kernel_index]

                # Now save correlation result in (create if not existent).
                # in our code: not every loop if, better calculate ahead of time & init at beginning
                # kernel_results shape: (num kernels, *correlation.shape)
                # Fill up the respective kernels
                if correlated_kernels is None:
                    correlated_kernels = np.zeros((self.num_kernels, *image_kernel_correlation.shape))
                correlated_kernels[kernel_index] = image_kernel_correlation


            # fill up output tensor for every processed image    
            if output_tensor is None:
                batch_size = input_tensor.shape[0]
                output_tensor = np.zeros((batch_size, *correlated_kernels.shape))
            output_tensor[batch_index] = correlated_kernels

        return output_tensor




        # 2D: [batch, channels, y coord, x coord]
        # 1D: [batch, channels, y coord]

        # loop over batch (input_tensor: first variable) -> channels & spacials remaining

            # loop over kernels in weights

                # eventuell 1kernel fall 1Dx1D handlen

                #IF: 2D Case
                    # n-dim correlate image & kernel
                    # something about channels???
                    # subsample by implementing the strides [::str_y. ::str_x] or other subsample func
                    # add bias of current kernel


                #ELIF: 1D Case
                    # same wie oben drüber nur mit subsampling in einer dim [::str_y]

                # Now save correlation result in (create if not existent).
                # in our code: not every loop if, better calculate ahead of time & init at beginning
                # kernel_results shape: (num kernels, *correlation.shape)
                # Fill up the respective kernels
            

            # init output tensor
            # shape: batchsize, *kernel_results
            # fill up output tensor for every processed image


        # return output tensor







    # gradient weights property
    @property
    def gradient_weights(self):  
        return  self._gradient_weights
    @gradient_weights.setter
    def gradient_weights(self,value):
        self._gradient_weights = value

    # gradient bias property
    @property
    def gradient_bias(self):  
        return  self._gradient_bias
    @gradient_bias.setter
    def gradient_bias(self,value):
        self._gradient_bias = value

    # optimizer property
    @property
    def optimizer(self):
        return self._optimizer
    @optimizer.setter
    def optimizer(self, optmizer):
        self._optimizer = copy.deepcopy(optmizer)
        self.bias_optimizer = copy.deepcopy(optmizer)
        # we need two deep copies since we want bias and weights to have separate optimizers








    def backward(self, error_tensor):

        # Channel 1 of error_tensor depends on channel 1 of all feature_maps (kernels)
        # weights: (feature_maps, input_channels,  x_coord, y_coord)
        # we want weights of the shape: input_channels, feature_maps, x_coord, y_coord) for easy convolution (since we used correlation in forward pass)

        # So we need to transpose weights from a,b,c(,d) to b,a,c(,d) -> flip along axis 1, 0, ...
        # reshape works a bit different (flatten to 1d and then redistribute. Not the same as transpose unfortunately in high dim)

        flipDirections = list(range(self.weights.ndim))
        flipDirections = [ 1, 0, *flipDirections[2:] ]
        weights = np.transpose(self.weights, axes=flipDirections)


        spacial_input_shape = self.input_tensor.shape[2:]


        output_tensor = None

        error_tensor_upsampled = np.zeros((*error_tensor.shape[:2], *spacial_input_shape))

        for img_batch_index, error_img_in_channel in enumerate(error_tensor): # loop over images within batch
            # error = error of image with index image_index within the whole batch

            

            upsampled_error = np.zeros((self.num_kernels, *spacial_input_shape))
            
            if self.is2D == True: 
                upsampled_error[:, ::self.stride_y, ::self.stride_x] = error_img_in_channel
            else:
                upsampled_error[:, ::self.stride_y] = error_img_in_channel

            error_tensor_upsampled[img_batch_index] = upsampled_error



            feature_maps = np.zeros((self.num_channels, *spacial_input_shape))

            for featureMap_number, conv_kernel in enumerate(weights):    # loop over kernels/feature maps
                # conv_kernel = weights associated to that kernel

                if self.is2D == True:
                    error_kernel_correlation = convolve(upsampled_error, conv_kernel[::-1], mode='constant', cval=0)[int((self.num_kernels)/2)]
                    feature_maps[featureMap_number] = error_kernel_correlation
                else:
                    error_kernel_correlation = convolve(upsampled_error, conv_kernel, mode='constant', cval=0)[
                        int((self.num_kernels) / 2)]
                    feature_maps[featureMap_number] = error_kernel_correlation

            if output_tensor is None:
                batch_size = error_tensor.shape[0]
                output_tensor = np.zeros((batch_size, *feature_maps.shape))
            output_tensor[img_batch_index] = feature_maps


        # now calc the bias gradient for each kernel -> vector!
        # now we want to get scalar errors for every kernel/feature_map (that is for all kernels together we want a vec of scalar errors)
        # thus we have to sum up over all axis but the featuremaps
        # eg error tensor in 2d case: image_index, feature_map_number, y_coord, x_coord -> ndim = 4 -> range -> 0,1,2,3 -> remove -> 0,2,3
        # 1d case:image_index, feature_map_number, y_coord -> ndim = 3 -> range -> 0,1,2 -> remove -> 0,2

        all_axis_but_kernels = list(range(error_tensor.ndim))
        all_axis_but_kernels.remove(1)
        all_axis_but_kernels = tuple(all_axis_but_kernels) # for np.sum ....

        gradient_bias = np.sum(error_tensor, axis=all_axis_but_kernels)
        self._gradient_bias = gradient_bias # save gradient




        # now calculate the gradient weights for each kernel (mutiple matrices / tensor !)

        gradient_weights = np.zeros((self.num_kernels, *self.convolution_shape))

       
        for img_batch_index, batch_img in enumerate(self.input_tensor): # loop over batch

            # now zero padding
            # 1/2 division because padding is supposed to be equally spaced around image
            # flor & ceil clear because we want padding to start outside of image
            # y is treaded same in both cases. only padding and x different for 1D and 2D

            # zero padding and not say interpolation because these places have no influence on forward pass
            # in backward they don't have contribution to loss -> so gradient is zero -> put in zeros

            y_start = int(np.floor((self.convolution_shape[1] - 1) / 2))
            y_stop = int(np.ceil((self.convolution_shape[1] - 1) / 2))

            if self.is2D == True:
                x_start = int(np.floor((self.convolution_shape[2] - 1) / 2))
                x_stop = int(np.ceil((self.convolution_shape[2] - 1) / 2))
                batch_img = np.pad(batch_img, ((0, 0), (y_start, y_stop), (x_start, x_stop)))

            else:
                batch_img = np.pad(batch_img, ((0, 0), (y_start, y_stop)))



            for channel_index, error_img_in_channel in enumerate(error_tensor_upsampled[img_batch_index]):     # loop over channels (eg RGB). Still 1D/2D error (=image)

                error_kernel_correlation = scipy.signal.correlate(batch_img, error_img_in_channel.reshape(1, *error_img_in_channel.shape), mode='valid') # convolve doesn't work??
                # ndim correlation doesn't support valid boundary somehow
                gradient_weights[channel_index] += error_kernel_correlation

            self._gradient_weights = gradient_weights   # save gradient


        # now call optimizer

        if self._optimizer:
            self.bias = self.bias_optimizer.calculate_update(self.bias, self._gradient_bias)
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)



        return output_tensor    # output_tensor[image_index, featuremap-weights]














    def initialize(self, weights_initializer, bias_initializer):
        
        self.weights = weights_initializer.initialize(self.weights_shape, self.fan_in, self.fan_out) 
        #self.bias = bias_initializer.initialize(self.num_kernels, self.num_kernels, 1) # Is this correct?
        self.bias = bias_initializer.initialize(self.num_kernels, self.fan_in, self.fan_out) # Is this correct?
        return