import numpy as np


class CrossEntropyLoss():
    # requires predictions to be greater than 1. Labels are expected to be {0,1}
    def __init__(self):
        self.prediction_tensor = None
        return

    def forward(self, input_tensor, label_tensor):

        '''Input: input_tensor, label_tensor
        Output: loss (scalar???)
        '''        

        log_value = np.log(input_tensor  + np.finfo(float).eps) # tutorial 1, slide 15. Elementwise log of predictions, eps for stability if 0
        loss = np.sum(label_tensor * -log_value) # Sum over the batch all instances where the label is 1
        self.prediction_tensor = input_tensor
        return loss 

    def backward(self,  label_tensor):
        error_tensor = - (label_tensor / self.prediction_tensor) # tutorial 1, slide 16
        return error_tensor    