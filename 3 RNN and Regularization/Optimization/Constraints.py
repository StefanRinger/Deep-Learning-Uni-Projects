import numpy as np


class L2_Regularizer:

    def __init__(self, alpha):
        # alpha represents the regularization weight
        self.alpha = alpha      # alpha = lagrangian multiplier. Used because labmda is python keyword
        


    def calculate_gradient(self, weights):

        # is called in backward pass. Optimizer incorporates this term into the overall update
        regularization_gradient = self.alpha * weights

        return regularization_gradient

    
    def norm(self, weights):    # for the forward pass.
        # Total loss: L_total (w) = L(w) + alpha * norm2(w)  -> sqrt(w)^2
        # This function only returns the second term: alpha * norm2(w)
        # Total loss has to be summed up in Neural Network Class

        regularization_loss = self.alpha * np.sum( weights ** 2 ) # sort of frobenius norm: square all mat elements, sum up and take sqrt.
        
        return regularization_loss
    







class L1_Regularizer:

    def __init__(self, alpha):
        # alpha represents the regularization weight
        self.alpha = alpha      # alpha = lagrangian multiplier. Used because labmda is python keyword
        


    def calculate_gradient(self, weights):

        # is called in backward pass. Optimizer incorporates this term into the overall update
        # Tutorial 3, page 5
        regularization_gradient = self.alpha * np.sign(weights)
        

        return regularization_gradient

    
    def norm(self, weights):    # for the forward pass.
        # Total loss: L_total (w) = L(w) + alpha * norm1(w)  -> sum(abs)
        # This function only returns the second term: alpha * norm1(w)
        # Total loss has to be summed up in Neural Network Class

        regularization_loss = self.alpha * np.sum( np.abs(weights) ) # abs norm = p=1

        return regularization_loss
    



