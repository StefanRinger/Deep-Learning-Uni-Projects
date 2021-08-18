
import numpy as np


class Optimizer:    # baseclass all optimizers inherit from

    regularizer = None

    def add_regularizer(self, regularizer):
        self.regularizer = regularizer
        return
    





class Sgd(Optimizer):
    def __init__(self ,  learning_rate: float):
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        updated_weight = weight_tensor - (self.learning_rate * gradient_tensor) # tutorial 1, slide 5: Gradient tensor = passed down error tensor
        

        if self.regularizer != None:
            regularization_gradient = self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)
            updated_weight = updated_weight - regularization_gradient # see optimizers/constraints.py Tutorial 3, page 4+5
        
        
        return updated_weight 

#Exercise 2 implementation



# includes history of gradient values into gradient
class SgdWithMomentum(Optimizer):
    #def __init__(self,learning_rate, momentum_rate) -> None:
    def __init__(self,learning_rate, momentum_rate=0.95):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.previousGradient = None

    def calculate_update(self, weight_tensor, gradient_tensor): # tut 2, slide 7.
        if self.previousGradient is None:   # for init
            self.previousGradient = np.zeros_like(gradient_tensor)


        momentum_Gradient = (self.momentum_rate * self.previousGradient) - (self.learning_rate * gradient_tensor) 
        
        # update the weights
        updated_weight =  weight_tensor  + momentum_Gradient


        if self.regularizer != None:
            regularization_gradient = self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)
            updated_weight = updated_weight - regularization_gradient # see optimizers/constraints.py Tutorial 3, page 4+5
        

        # save the current gradient for the next iteration as old gradient
        self.previousGradient = momentum_Gradient 
        return updated_weight





# assigns individual learning rates to weights based on how much their gradients oscillate
class Adam(Optimizer):
    def __init__(self, learning_rate = 0.001 , mu = 0.9, rho = 0.999):

        #   g = gradient tensor; v = currentLayerMomentum ; r = currentLayerR ; k = exponential
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.previousV = None
        self.previousR = None
        self.k = 1  # why is no value given for this???

    def calculate_update(self, weight_tensor, gradient_tensor): # Tutorial 2, Slide 9 

        if self.previousV is None:
            self.previousV = np.zeros_like(gradient_tensor)
        if self.previousR is None:
            self.previousR = np.zeros_like(gradient_tensor)    

        # introduce the momentum like variables V and R: 
        currentV = (self.mu * self.previousV) + (1 - self.mu) * gradient_tensor
        currentR = (self.rho * self.previousR) + (1- self.rho) *  gradient_tensor * gradient_tensor # elementwise product

        # Correct V and R for their bias:
        biasCorrectedV = currentV / (1 - self.mu ** self.k) # ** is exponenet
        biasCorrectedR = currentR / (1 - self.rho ** self.k)

        # update the weights
        updated_weight =  weight_tensor - self.learning_rate * (biasCorrectedV / (np.sqrt(biasCorrectedR) + np.finfo('float').eps ))


        if self.regularizer != None:
            regularization_gradient = self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)
            updated_weight = updated_weight - regularization_gradient # see optimizers/constraints.py Tutorial 3, page 4+5
        

        # save the current V and R for the next iteration
        self.previousV = currentV
        self.previousR = currentR
        self.k += 1 # k is time index and has to be incremented

        return updated_weight 
