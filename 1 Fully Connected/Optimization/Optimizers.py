
import numpy as np

class Sgd:
    def __init__(self ,  learning_rate: float):
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        updated_weight = weight_tensor - (self.learning_rate * gradient_tensor) # tutorial 1, slide 5: Gradient tensor = passed down error tensor
        return updated_weight 
